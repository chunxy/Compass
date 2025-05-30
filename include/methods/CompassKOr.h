#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <queue>
#include <utility>
#include <vector>
#include "Pod.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/Compass.h"
#include "utils/predicate.h"

using std::pair;
using std::priority_queue;
using std::vector;

// Old racing version, not used.
template <typename dist_t, typename attr_t>
class CompassKOr : public Compass<dist_t, attr_t> {
 protected:
  faiss::IndexIVFFlat *ivf_flat_;

 public:
  CompassKOr(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist)
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist),
        ivf_flat_(new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist)) {
    this->ivf_ = dynamic_cast<faiss::Index *>(ivf_flat_);
  }

  void AssignPoints(const size_t n, const dist_t *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    if (distances == nullptr) {
      ivf_flat_->quantizer->assign(n, (float *)data, assigned_clusters, k);
    } else {
      ivf_flat_->quantizer->search(n, (float *)data, k, distances, assigned_clusters);
    }
  }

  // By default, we will not use the distances to centroids.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const dist_t *query,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) override {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;
    AssignPoints(nq, query, nprobe, this->query_cluster_rank_, nullptr);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(this->n_, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;

      std::vector<std::unordered_set<labeltype>> candidates_per_dim(this->da_);
      for (int j = 0; j < this->da_; ++j) {
        auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][j];
        auto itr_beg = btree.lower_bound(l_bound[j]);
        auto itr_end = btree.upper_bound(u_bound[j]);
        for (auto itr = itr_beg; itr != itr_end; ++itr) {
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
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1))) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                std::vector<std::unordered_set<labeltype>> candidates_per_dim(this->da_);
                for (int j = 0; j < this->da_; ++j) {
                  auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][j];
                  auto itr_beg = btree.lower_bound(l_bound[j]);
                  auto itr_end = btree.upper_bound(u_bound[j]);
                  for (auto itr = itr_beg; itr != itr_end; ++itr) {
                    candidates_per_dim[j].insert(itr->second);
                  }
                }
                // Intersect all sets in candidates_per_dim
                std::unordered_set<labeltype> intersection;
                if (this->da_ > 0) intersection = candidates_per_dim[0];
                for (int j = 1; j < this->da_; j++) {
                  std::unordered_set<labeltype> temp;
                  for (const auto &id : intersection) {
                    if (candidates_per_dim[j].count(id)) {
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
            _mm_prefetch(this->hnsw_.getDataByInternalId(*itr_beg), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_((float *)query + q * this->d_, vect, this->hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          metrics[q].nround++;
          int cnt = this->hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second]) continue;
            visited[top.second] = true;
            metrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        this->hnsw_.ReentrantSearchKnn(
            (float *)query + q * this->d_,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }
};
