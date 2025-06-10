#pragma once

#include <algorithm>
#include "utils/predicate.h"
#include "Compass1dK.h"

// Use graph to find entry points.
template <typename dist_t, typename attr_t>
class Compass1dKGr : public Compass1dK<dist_t, attr_t> {
 public:
  Compass1dKGr(size_t n, size_t d, size_t M, size_t efc, size_t nlist)
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
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;
    this->SearchClusters(nq, query, nprobe, this->query_cluster_rank_);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, 1);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;

      vector<bool> visited(this->hnsw_.cur_element_count, false);

      {
        tableint currObj = this->hnsw_.enterpoint_node_;
        dist_t curdist = this->hnsw_.fstdistfunc_(
            (float *)query + q * this->d_,
            this->hnsw_.getDataByInternalId(this->hnsw_.enterpoint_node_),
            this->hnsw_.dist_func_param_
        );

        for (int level = this->hnsw_.maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)this->hnsw_.get_linklist(currObj, level);
            int size = this->hnsw_.getListCount(data);
            metrics[q].ncomp += size;

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > this->hnsw_.max_elements_) throw std::runtime_error("cand error");
              dist_t d = this->hnsw_.fstdistfunc_(
                  (float *)query + q * this->d_, this->hnsw_.getDataByInternalId(cand), this->hnsw_.dist_func_param_
              );

              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
        // ranked_clusters = ranked_clusters_ + currObj * nprobe;
        visited[currObj] = true;
        candidate_set.emplace(-curdist, currObj);
        if (pred(currObj)) top_candidates.emplace(curdist, currObj);
        metrics[q].cand_dist.push_back(curdist);
      }

      auto curr_ci = q * nprobe;
      auto itr_beg = this->btrees_[this->query_cluster_rank_[curr_ci]].lower_bound(*l_bound);
      auto itr_end = this->btrees_[this->query_cluster_rank_[curr_ci]].upper_bound(*u_bound);

      while (true) {
        int crel = 0;
        // if (candidate_set.empty() || distances[curr_ci] <
        // -candidate_set.top().first) {
        if (candidate_set.empty() ||
            (!top_candidates.empty() && -candidate_set.top().first > top_candidates.top().first)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              if (curr_ci + 1 >= (q + 1) * nprobe)
                break;
              else {
                curr_ci++;
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
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_((float *)query + q * this->d_, vect, this->hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            if (top_candidates.size() < efs || dist < upper_bound) {
              candidate_set.emplace(-dist, tableid);
              top_candidates.emplace(dist, tableid);
              metrics[q].is_ivf_ppsl[tableid] = true;
              if (top_candidates.size() > efs_) top_candidates.pop();
            }
          }
          metrics[q].nround++;
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
