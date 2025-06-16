#pragma once

#include <algorithm>
#include "Compass1dK.h"
#include "hnswlib/hnswlib.h"
#include "utils/predicate.h"

// Old racing version, not used.
template <typename dist_t, typename attr_t>
class Compass1dKOr : public Compass1dK<dist_t, attr_t> {
 public:
  Compass1dKOr(size_t n, size_t d, size_t M, size_t efc, size_t nlist)
      : Compass1dK<dist_t, attr_t>(n, d, M, efc, nlist) {}

  // For ranking clusters using IVF.
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
      BatchMetric &bm
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;
    this->SearchClusters(nq, query, nprobe, this->query_cluster_rank_, bm, this->distances_);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, 1);
    VisitedList* vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vl->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->hnsw_.cur_element_count, false);

      int curr_ci = q * nprobe;
      auto itr_beg = this->btrees_[this->query_cluster_rank_[curr_ci]].lower_bound(*l_bound);
      auto itr_end = this->btrees_[this->query_cluster_rank_[curr_ci]].upper_bound(*u_bound);

      while (true) {
        int crel = 0;
        if (candidate_set.empty() ||
            (curr_ci < nprobe * (q + 1) && -candidate_set.top().first > this->distances_[curr_ci])) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                itr_beg = this->btrees_[this->query_cluster_rank_[curr_ci]].lower_bound(*l_bound);
                itr_end = this->btrees_[this->query_cluster_rank_[curr_ci]].upper_bound(*u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(this->hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;
            visited[tableid] = visited_tag;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_((float *)query + q * this->d_, vect, this->hnsw_.dist_func_param_);
            bm.qmetrics[q].ncomp++;
            bm.qmetrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            candidate_set.emplace(-dist, tableid);
            if (dist < upper_bound) {
              top_candidates.emplace(dist, tableid);
              if (top_candidates.size() > efs_) top_candidates.pop();
            } else {
              recycle_set.emplace(-dist, tableid);
            }
          }
          bm.qmetrics[q].nround++;
        }

        this->hnsw_.ReentrantSearchKnn(
            (float *)query + q * this->d_,
            k,
            -1,
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
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }
};
