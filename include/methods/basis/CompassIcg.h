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
      size_t delta_efs
  )
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist) {
    this->isearch_ = new IterativeSearch<cg_dist_t>(n, d, s, M_cg);
    this->isearch_->SetSearchParam(batch_k, delta_efs);
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
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;

    const void *query, *xquery;
    if (std::holds_alternative<const void *>(var)) {
      query = std::get<const void *>(var);
      xquery = query;
    } else {
      query = std::get<pair<const void *, const void *>>(var).first;
      xquery = std::get<pair<const void *, const void *>>(var).second;
    }
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, 1);
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = std::move(Open(xquery, q, nprobe));

      auto next = isearch_->Next(&state);
      int clus = next.second, clus_cnt = 1;

      std::vector<std::unordered_set<labeltype>> candidates_per_dim(this->da_);
      for (int j = 0; j < this->da_; ++j) {
        auto &btree = this->btrees_[clus][j];
        auto beg = btree.lower_bound(l_bound[j]);
        auto end = btree.upper_bound(u_bound[j]);
        for (auto itr = beg; itr != end; ++itr) {
          candidates_per_dim[j].insert(itr->second);
        }
      }
      // Intersect all sets in candidates_per_dim
      std::unordered_set<labeltype> intersection;
      if (this->da_ > 0) intersection = candidates_per_dim[0];
      for (int j = 1; j < this->da_; ++j) {
        std::unordered_set<labeltype> temp;
        for (const auto &id : intersection) {
          if (candidates_per_dim[j].count(id)) {
            temp.insert(id);
          }
        }
        intersection = std::move(temp);
      }

      auto itr_beg = intersection.begin();
      auto itr_end = intersection.end();

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
                std::vector<std::unordered_set<labeltype>> _candidates_per_dim(this->da_);
                for (int j = 0; j < this->da_; ++j) {
                  auto &btree = this->btrees_[clus][j];
                  auto beg = btree.lower_bound(l_bound[j]);
                  auto end = btree.upper_bound(u_bound[j]);
                  for (auto itr = beg; itr != end; ++itr) {
                    _candidates_per_dim[j].insert(itr->second);
                  }
                }
                // Intersect all sets in candidates_per_dim
                if (this->da_ > 0) intersection = _candidates_per_dim[0];
                for (int j = 1; j < this->da_; ++j) {
                  std::unordered_set<labeltype> temp;
                  for (const auto &id : intersection) {
                    if (_candidates_per_dim[j].count(id)) {
                      temp.insert(id);
                    }
                  }
                  intersection = std::move(temp);
                }
                itr_beg = intersection.begin();
                itr_end = intersection.end();
                continue;
              }
            }

            auto tableid = *itr_beg;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(this->hnsw_.getDataByInternalId(tableid), _MM_HINT_T0);
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

      bm.cluster_search_ncomp += isearch_->GetNcomp(&state);
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