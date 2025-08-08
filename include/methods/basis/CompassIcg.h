#pragma once

#include <type_traits>
#include "Compass.h"
#include "faiss/MetricType.h"
#include "methods/basis/IterativeSearch.h"

using hnswlib::L2Space;
using hnswlib::L2SpaceB;

template <typename dist_t, typename attr_t, typename cg_dist_t = float>
class CompassIcg : public Compass<dist_t, attr_t> {
 protected:
  IterativeSearch<cg_dist_t> *isearch_;

 protected:
  void SearchClusters(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) override {}  // dummy implementation

  virtual IterativeSearchState<cg_dist_t> Open(const void *query, int idx, int nprobe) {
    const void *target = ((char *)query) + this->isearch_->hnsw_->data_size_ * idx;
    return this->isearch_->Open(target, nprobe);
  }

  virtual IterativeSearchState<cg_dist_t> Open(const void *query, int idx, int nprobe, VisitedList *vl) {
    const void *target = ((char *)query) + this->isearch_->hnsw_->data_size_ * idx;
    return this->isearch_->Open(target, nprobe, vl);
  }

  virtual const void *icg_transform(const void *query, int nq) { return query; }

 public:
  CompassIcg(
      size_t n,
      size_t d,
      SpaceInterface<cg_dist_t> *s,
      size_t da,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t initial_efs,
      size_t delta_efs
  )
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist) {
    this->isearch_ = new IterativeSearch<cg_dist_t>(n, d, s, M_cg);
    this->isearch_->SetSearchParam(batch_k, initial_efs, delta_efs);
  }

  // By default, we will not use the distances to centroids.
  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnn(
      const std::variant<const void *, pair<const void *, const void *>> &var,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) override {
    auto batch_start = std::chrono::high_resolution_clock::system_clock::now();
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_;

    const void *query, *xquery;
    if (std::holds_alternative<const void *>(var)) {
      query = std::get<const void *>(var);
      xquery = this->icg_transform(query, nq);
    } else {
      query = std::get<pair<const void *, const void *>>(var).first;
      xquery = std::get<pair<const void *, const void *>>(var).second;
    }
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->isearch_->hnsw_->visited_list_pool_->getFreeVisitedList();

    auto batch_stop = std::chrono::high_resolution_clock::system_clock::now();
    auto batch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(batch_stop - batch_start).count();
    bm.overhead += batch_time;

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      auto query_start = std::chrono::high_resolution_clock::system_clock::now();
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_cg->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = std::move(Open(xquery, q, nprobe, vl_cg));

      auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
      auto next = isearch_->Next(&state);
      auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
      bm.qmetrics[q].cg_latency += cg_time;

      int clus = next.second, clus_cnt = 1;

      auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
      auto itr_beg = this->btrees_[clus].lower_bound(l_bound[0]);
      auto itr_end = this->btrees_[clus].upper_bound(u_bound[0]);
      while (itr_beg != itr_end && !pred(itr_beg->second.second)) {
        itr_beg++;
      }
      auto btree_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_stop - btree_start).count();
      bm.qmetrics[q].btree_latency += btree_time;

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (clus != -1)) {
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
              auto _next = isearch_->Next(&state);
              auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
              auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
              bm.qmetrics[q].cg_latency += cg_time;
              clus = _next.second;
              clus_cnt++;
              if (clus == -1)
                break;
              else {
                auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
                itr_beg = this->btrees_[clus].lower_bound(l_bound[0]);
                itr_end = this->btrees_[clus].upper_bound(u_bound[0]);
                while (itr_beg != itr_end && !pred(itr_beg->second.second)) {
                  itr_beg++;
                }
                auto btree_stop = std::chrono::high_resolution_clock::system_clock::now();
                auto btree_time =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(btree_stop - btree_start).count();
                bm.qmetrics[q].btree_latency += btree_time;
                continue;
              }
            }

            auto tableid = itr_beg->second.first;
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
            do {
              itr_beg++;
            } while (itr_beg != itr_end && !pred(itr_beg->second.second));
            auto btree_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_stop - btree_start).count();
            bm.qmetrics[q].btree_latency += btree_time;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->hnsw_.getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_(
                (char *)query + this->hnsw_.data_size_ * q, vect, this->hnsw_.dist_func_param_
            );
            bm.qmetrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
          bm.qmetrics[q].nround++;
          int cnt = this->hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second] == visited_tag) continue;
            visited[top.second] = visited_tag;
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
        this->hnsw_.ReentrantSearchKnnBounded(
            (char *)query + this->hnsw_.data_size_ * q,
            k,
            -recycle_set.top().first,  // cause infinite loop?
            // distances[curr_ci],
            top_candidates,
            candidate_set,
            vl,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
        auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
        bm.qmetrics[q].graph_latency += graph_time;
        if ((top_candidates.size() >= efs_) || clus == -1) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = clus_cnt;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      bm.qmetrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);

      auto query_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto query_time = std::chrono::duration_cast<std::chrono::nanoseconds>(query_stop - query_start).count();
      bm.qmetrics[q].latency += query_time;
      bm.qmetrics[q].ncomp_cg += isearch_->GetNcomp(&state);
      isearch_->Close(&state);
    }

    return results;
  }

  // By default, we will not use the distances to centroids.
  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnTwoQueues(
      const std::variant<const void *, pair<const void *, const void *>> &var,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_;

    const void *query, *xquery;
    if (std::holds_alternative<const void *>(var)) {
      query = std::get<const void *>(var);
      xquery = this->icg_transform(query, nq);
    } else {
      query = std::get<pair<const void *, const void *>>(var).first;
      xquery = std::get<pair<const void *, const void *>>(var).second;
    }
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->isearch_->hnsw_->visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_cg->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = std::move(Open(xquery, q, nprobe, vl_cg));

      auto next = isearch_->Next(&state);
      int clus = next.second, clus_cnt = 1;

      auto itr_beg = this->btrees_[clus].lower_bound(l_bound[0]);
      auto itr_end = this->btrees_[clus].upper_bound(u_bound[0]);
      while (itr_beg != itr_end && !pred(itr_beg->second.second)) {
        itr_beg++;
      }

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (clus != -1)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              auto _next = isearch_->Next(&state);
              clus = _next.second;
              clus_cnt++;
              if (clus == -1)
                break;
              else {
                itr_beg = this->btrees_[clus].lower_bound(l_bound[0]);
                itr_end = this->btrees_[clus].upper_bound(u_bound[0]);
                while (itr_beg != itr_end && !pred(itr_beg->second.second)) {
                  itr_beg++;
                }
                continue;
              }
            }

            auto tableid = itr_beg->second.first;
            do {
              itr_beg++;
            } while (itr_beg != itr_end && !pred(itr_beg->second.second));
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->hnsw_.getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_(
                (char *)query + this->hnsw_.data_size_ * q, vect, this->hnsw_.dist_func_param_
            );
            bm.qmetrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
            visited[tableid] = visited_tag;
            bm.qmetrics[q].is_ivf_ppsl[tableid] = true;
            candidate_set.emplace(-dist, tableid);
          }
          bm.qmetrics[q].nround++;
        }

        this->hnsw_.ReentrantSearchKnnBounded(
            (char *)query + this->hnsw_.data_size_ * q,
            k,
            top_candidates,
            recycle_set,
            candidate_set,
            vl,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || clus == -1) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = clus_cnt;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (recycle_set.size() > k) recycle_set.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        top_candidates.emplace(top.first, top.second);
        if (top_candidates.size() > k) top_candidates.pop();
        recycle_set.pop();
      }
      bm.qmetrics[q].nrecycled = nrecycled;
      results[q] = std::move(top_candidates);

      bm.qmetrics[q].ncomp_cg = isearch_->GetNcomp(&state);
      isearch_->Close(&state);
    }

    return results;
  }

  // By default, we will not use the distances to centroids.
  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnRtree(
      const std::variant<const void *, pair<const void *, const void *>> &var,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_;

    const void *query, *xquery;
    if (std::holds_alternative<const void *>(var)) {
      query = std::get<const void *>(var);
      xquery = this->icg_transform(query, nq);
    } else {
      query = std::get<pair<const void *, const void *>>(var).first;
      xquery = std::get<pair<const void *, const void *>>(var).second;
    }
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->isearch_->hnsw_->visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_cg->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = std::move(Open(xquery, q, nprobe, vl_cg));

      auto next = isearch_->Next(&state);
      int clus = next.second, clus_cnt = 1;

      point min_corner(l_bound[0], l_bound[1]), max_corner(u_bound[0], u_bound[1]);
      box b(min_corner, max_corner);
      auto itr_beg = this->rtrees_[clus].qbegin(geo::index::covered_by(b));
      auto itr_end = this->rtrees_[clus].qend();

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (clus != -1)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              auto _next = isearch_->Next(&state);
              clus = _next.second;
              clus_cnt++;
              if (clus == -1)
                break;
              else {
                itr_beg = this->rtrees_[clus].qbegin(geo::index::covered_by(b));
                itr_end = this->rtrees_[clus].qend();
                continue;
              }
            }

            auto tableid = itr_beg->second;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->hnsw_.getDataByInternalId(itr_beg->second), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_(
                (char *)query + this->hnsw_.data_size_ * q, vect, this->hnsw_.dist_func_param_
            );
            bm.qmetrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          bm.qmetrics[q].nround++;
          int cnt = this->hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second] == visited_tag) continue;
            visited[top.second] = visited_tag;
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        this->hnsw_.ReentrantSearchKnnBounded(
            (char *)query + this->hnsw_.data_size_ * q,
            k,
            -recycle_set.top().first,  // cause infinite loop?
            // distances[curr_ci],
            top_candidates,
            candidate_set,
            vl,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || clus == -1) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = clus_cnt;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      bm.qmetrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);

      bm.qmetrics[q].ncomp_cg = isearch_->GetNcomp(&state);
      isearch_->Close(&state);
    }

    return results;
  }

  // By default, we will not use the distances to centroids.
  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnCtree(
      const std::variant<const void *, pair<const void *, const void *>> &var,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_;

    const void *query, *xquery;
    if (std::holds_alternative<const void *>(var)) {
      query = std::get<const void *>(var);
      xquery = this->icg_transform(query, nq);
    } else {
      query = std::get<pair<const void *, const void *>>(var).first;
      xquery = std::get<pair<const void *, const void *>>(var).second;
    }
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    array<attr_t, 4> l_bound_arr{-1, -1, -1, -1};
    array<attr_t, 4> u_bound_arr{10001, 10001, 10001, 10001};
    for (int i = 0; i < this->da_; i++) {
      l_bound_arr[i] = l_bound[i];
      u_bound_arr[i] = u_bound[i];
    }
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->isearch_->hnsw_->visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_cg->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = std::move(Open(xquery, q, nprobe, vl_cg));

      auto next = isearch_->Next(&state);
      int clus = next.second, clus_cnt = 1;

      auto itr_beg = this->ctrees_[clus].lower_bound(l_bound_arr);
      auto itr_end = this->ctrees_[clus].upper_bound(u_bound_arr);
      while (itr_beg != itr_end && !pred(itr_beg->first)) {
        itr_beg++;
      }

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (clus != -1)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              auto _next = isearch_->Next(&state);
              clus = _next.second;
              clus_cnt++;
              if (clus == -1)
                break;
              else {
                itr_beg = this->ctrees_[clus].lower_bound(l_bound_arr);
                itr_end = this->ctrees_[clus].upper_bound(u_bound_arr);
                while (itr_beg != itr_end && !pred(itr_beg->first)) {
                  itr_beg++;
                }
                continue;
              }
            }

            auto tableid = itr_beg->second;
            do {
              itr_beg++;
            } while (itr_beg != itr_end && !pred(itr_beg->first));
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->hnsw_.getDataByInternalId(itr_beg->second), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_(
                (char *)query + this->hnsw_.data_size_ * q, vect, this->hnsw_.dist_func_param_
            );
            bm.qmetrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          bm.qmetrics[q].nround++;
          // int cnt = this->hnsw_.M_;
          int cnt = crel / 10;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second] == visited_tag) continue;
            visited[top.second] = visited_tag;
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        this->hnsw_.ReentrantSearchKnnBounded(
            (char *)query + this->hnsw_.data_size_ * q,
            k,
            -recycle_set.top().first,  // cause infinite loop?
            // distances[curr_ci],
            top_candidates,
            candidate_set,
            vl,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || clus == -1) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = clus_cnt;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      bm.qmetrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);

      bm.qmetrics[q].ncomp_cg = isearch_->GetNcomp(&state);
      isearch_->Close(&state);
    }

    return results;
  }

  virtual void BuildClusterGraph() = 0;

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->isearch_->hnsw_->saveIndex(path.string());
  }

  virtual void LoadClusterGraph(fs::path path) {
    using SpaceType = typename std::conditional<std::is_same<cg_dist_t, int>::value, L2SpaceB, L2Space>::type;
    this->isearch_->hnsw_->loadIndex(path.string(), new SpaceType(this->d_));
  }
};