#pragma once

#include <algorithm>
#include "../hnswlib/hnswlib.h"
#include "../utils/predicate.h"
#include "Compass1d.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

template <typename dist_t, typename attr_t>
class Compass1dXCg : public Compass1d<dist_t, attr_t> {
  HierarchicalNSW<dist_t> cgraph_;

 public:
  Compass1dXCg(size_t n, size_t d, size_t M, size_t efc, size_t nlist, size_t dx)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist) {
    this->ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
  }

  // Dummy implementation.
  void AssignPoints(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) override {};

  // For ranking clusters using graph.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const void *xquery,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;
      auto clusters = this->cgraph_.searchKnnCloserFirst((float *)(xquery) + q * this->d_, nprobe);

      vector<bool> visited(this->n_, false);

      RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, 1);
      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = 0;
      auto itr_beg = this->btrees_[clusters[curr_ci].second].lower_bound(l_bound);
      auto itr_end = this->btrees_[clusters[curr_ci].second].upper_bound(u_bound);

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= nprobe)
                break;
              else {
                itr_beg = this->btrees_[clusters[curr_ci].second].lower_bound(l_bound);
                itr_end = this->btrees_[clusters[curr_ci].second].upper_bound(u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(this->hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
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

        this->hnsw_.ReentrantSearchKnnBounded(
            (float *)query + q * this->d_,
            k,
            -recycle_set.top().first,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || curr_ci >= nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          metrics[q].is_ivf_ppsl[top.second] = true;
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

  void BuildClusterGraph() {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->cgraph_.addPoint(centroids + i * ivf_flat->d, i);
    }
  }

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cgraph_.saveIndex(path.string());
  }

  void LoadClusterGraph(fs::path path) { this->cgraph_.loadIndex(path.string(), new L2Space(this->d_)); }
};
