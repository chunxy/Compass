#pragma once

#include <algorithm>
#include <cstddef>
#include "../hnswlib/hnswlib.h"
// #include "faiss/MetricType.h"

using namespace hnswlib;

template <typename dist_t>
class ReentrantHNSW : public HierarchicalNSW<dist_t> {
 public:
  ReentrantHNSW(
      SpaceInterface<dist_t> *s,
      size_t max_elements,
      size_t M = 16,
      size_t efc = 200,
      size_t random_seed = 100,
      bool allow_replace_deleted = false
  )
      : HierarchicalNSW<dist_t>(s, max_elements, M, efc, random_seed, allow_replace_deleted) {}

  void ReentrantSearchKnn(
      const void *query_data,
      size_t k,
      int nhops,
      std::priority_queue<std::pair<dist_t, int64_t>> &top_candidates,
      std::priority_queue<std::pair<dist_t, int64_t>> &candidate_set,
      std::vector<bool> &visited,
      BaseFilterFunctor *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound =
        top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      // if (curr_obj < 0) break;
      candidate_set.pop();

      if (curr_dist > upper_bound && top_candidates.size() >= efs) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      // this->metric_hops++;
      // this->metric_distance_computations += size;
      tableint *cand_nbrs = (tableint *)(cand_info + 1);

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
        if (visited[cand_nbr]) continue;
        visited[cand_nbr] = true;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(this->getExternalLabel(cand_nbr))) continue;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          is_graph_ppsl[cand_nbr] = true;
          if (top_candidates.size() > efs) top_candidates.pop();
          upper_bound = top_candidates.top().first;
        }

      }
    }
  }
};