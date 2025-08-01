#pragma once

#include <variant>
#include "HybridIndex.h"
#include "btree_map.h"
#include "utils/predicate.h"

using hnswlib::labeltype;
using std::array;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass : public HybridIndex<dist_t, attr_t> {
 protected:
  vector<btree::btree_map<attr_t, pair<labeltype, array<attr_t, 4>>>> btrees_;
  int da_;

  // Assign original/transformed points to clusters.
  // Called during index building.
  virtual void AssignPoints(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) = 0;

  // Potentially assign original/transformed points to clusters using cluster graph,
  // as well as profile the initial search process.
  virtual void SearchClusters(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) {
    auto assign_beg = std::chrono::high_resolution_clock::now();
    AssignPoints(n, data, k, assigned_clusters, distances);
    auto assign_end = std::chrono::high_resolution_clock::now();
    bm.cluster_search_time = std::chrono::duration_cast<std::chrono::microseconds>(assign_end - assign_beg).count();
  }

 public:
  Compass(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist)
      : HybridIndex<dist_t, attr_t>(n, d, M, efc, nlist),
        btrees_(nlist, btree::btree_map<attr_t, pair<labeltype, array<attr_t, 4>>>()),
        da_(da) {}

  void AddPointsToIvf(const size_t n, const void *data, const labeltype *labels, const attr_t *attrs) override {
    AssignPoints(n, data, 1, this->base_cluster_rank_);
    for (int i = 0; i < n; i++) {
      // vector<attr_t> arr(da_);
      array<attr_t, 4> arr;
      for (int j = 0; j < da_; j++) {
        arr[j] = attrs[i * da_ + j];
      }
      btrees_[this->base_cluster_rank_[i]].insert(
          std::make_pair(attrs[i * da_], std::make_pair(labels[i], std::move(arr)))
      );
    }
  }

  void LoadRanking(fs::path path, attr_t *attrs) override {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < this->n_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      // vector<attr_t> arr(da_);
      array<attr_t, 4> arr;
      int a = sizeof(arr);
      for (int j = 0; j < da_; j++) {
        arr[j] = attrs[i * da_ + j];
      }
      btrees_[assigned_cluster].insert(std::make_pair(attrs[i * da_], std::make_pair(i, std::move(arr))));
    }
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
    SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
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

      bm.qmetrics[q].nround = 0;
      bm.qmetrics[q].ncomp = 0;

      int curr_ci = q * nprobe;

      // std::unordered_set<labeltype> intersection;
      // {
      //   auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][0];
      //   auto beg = btree.lower_bound(l_bound[0]);
      //   auto end = btree.upper_bound(u_bound[0]);
      //   for (auto itr = beg; itr != end; ++itr) {
      //     if (pred(itr->second)) {
      //       intersection.insert(itr->second);
      //     }
      //   }
      // }

      auto itr_beg = this->btrees_[this->query_cluster_rank_[curr_ci]].lower_bound(l_bound[0]);
      auto itr_end = this->btrees_[this->query_cluster_rank_[curr_ci]].upper_bound(u_bound[0]);
      while (itr_beg != itr_end && !pred(itr_beg->second.second)) {
        itr_beg++;
      }

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1))) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                // std::unordered_set<labeltype> _intersection;
                // {
                //   auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][0];
                //   auto beg = btree.lower_bound(l_bound[0]);
                //   auto end = btree.upper_bound(u_bound[0]);
                //   for (auto itr = beg; itr != end; ++itr) {
                //     if (pred(itr->second)) {
                //       _intersection.insert(itr->second);
                //     }
                //   }
                // }
                // intersection = std::move(_intersection);
                itr_beg = this->btrees_[this->query_cluster_rank_[curr_ci]].lower_bound(l_bound[0]);
                itr_end = this->btrees_[this->query_cluster_rank_[curr_ci]].upper_bound(u_bound[0]);
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
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = curr_ci - q * nprobe;
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
    }

    return results;
  }
};