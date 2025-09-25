#pragma once

#include <algorithm>
#include <cstddef>
#include "avl.h"
#include "fc/btree.h"
#include "hnswlib/hnswlib.h"
#include "utils/out.h"
#include "utils/predicate.h"

namespace fc = frozenca;

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

  ReentrantHNSW(
      SpaceInterface<dist_t> *s,
      const std::string &location,
      bool nmslib = false,
      size_t max_elements = 0,
      bool allow_replace_deleted = false
  )
      : HierarchicalNSW<dist_t>(s, location, nmslib, max_elements, allow_replace_deleted) {}

  void IterativeReentrantSearchKnn(
      const void *query_data,
      const size_t k,
      std::priority_queue<std::pair<dist_t, int64_t>> &recycled_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);

        result_set.emplace(-cand_nbr_dist, cand_nbr);
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) {
            auto top = top_candidates.top();
            recycled_candidates.emplace(-top.first, top.second);
            top_candidates.pop();
          }
          upper_bound = top_candidates.top().first;
        } else {
          recycled_candidates.emplace(-cand_nbr_dist, -(int64_t)cand_nbr);  // mark as directly dropped
        }
      }
    }
  }

  void IterativeReentrantSearchKnnPostFiltered(
      const void *query_data,
      const size_t k,
      BaseFilterFunctor *is_id_allowed,
      std::priority_queue<std::pair<dist_t, int64_t>> &recycled_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      int satisfied_count = 0;
      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);

        if (is_id_allowed == nullptr || (*is_id_allowed)(cand_nbr)) {
          result_set.emplace(-cand_nbr_dist, cand_nbr);
          satisfied_count++;
        }
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) {
            auto top = top_candidates.top();
            recycled_candidates.emplace(-top.first, top.second);
            top_candidates.pop();
          }
          upper_bound = top_candidates.top().first;
        } else {
          recycled_candidates.emplace(-cand_nbr_dist, -(int64_t)cand_nbr);
        }
      }

      // This incurs too many computations.
      // Two-hop search in post filtering does not work.
      if (float(satisfied_count) / size <= 0.1) {
        for (int i = 0; i < size; i++) {
          tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
          _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif

          tableint *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;

          for (int j = 0; j < twohop_size; j++) {
            tableint twohop_nbr = twohop_nbrs[j];

            if (vl->mass[twohop_nbr] == vl->curV) continue;
            vl->mass[twohop_nbr] = vl->curV;
            ncomp++;
            dist_t twohop_nbr_dist =
                this->fstdistfunc_(query_data, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);

            if (is_id_allowed == nullptr || (*is_id_allowed)(twohop_nbr)) {
              result_set.emplace(-twohop_nbr_dist, twohop_nbr);
              satisfied_count++;
            }
            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);
#ifdef USE_SSE
              _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
              top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              }
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-twohop_nbr_dist, -(int64_t)twohop_nbr);
            }
          }
        }
      }
    }
  }

  void IterativeReentrantSearchKnnTwoHop(
      const void *query_data,
      const size_t k,
      RangeQuery<float> *is_id_allowed,
      std::priority_queue<std::pair<dist_t, int64_t>> &recycled_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp,
      float &sel
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    int total_added_count = 0, total_checked_count = 0;
    while (!candidate_set.empty()) {
      int added_count = 0, checked_count = 0, satisfied_count = 0;
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch((char *)(is_id_allowed->prefetch(*(cand_nbrs))), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      std::priority_queue<std::pair<dist_t, tableint>> added_onehop_neighbors;
      std::set<tableint> other_onehop_neighbors;
      // It may occur that the visited count is 0, which is not expected.
      // Work around this first.

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(is_id_allowed->prefetch(*(cand_nbrs + i + 1))), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (is_id_allowed == nullptr || (*is_id_allowed)(cand_nbr)) {
          satisfied_count++;
        }
        if (vl->mass[cand_nbr] == vl->curV) continue;
        checked_count++;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) {
          other_onehop_neighbors.insert(cand_nbr);
          continue;
        };
        vl->mass[cand_nbr] = vl->curV;

        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);

        result_set.emplace(-cand_nbr_dist, cand_nbr);
        added_count++;
        added_onehop_neighbors.emplace(-cand_nbr_dist, cand_nbr);

        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          // No need to put on candidate_set, because it will be traversed later.
          // candidate_set.emplace(-cand_nbr_dist, cand_nbr);
          // #ifdef USE_SSE
          //           _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
          // #endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) {
            auto top = top_candidates.top();
            recycled_candidates.emplace(-top.first, top.second);
            top_candidates.pop();
          }
          upper_bound = top_candidates.top().first;
        } else {
          recycled_candidates.emplace(-cand_nbr_dist, -(int64_t)cand_nbr);
        }
      }

      // TODO: The ratio is a tuning point.
      // adaptive local
      // float selectivity = checked_count == 0 ? 0 : (float)added_count / checked_count;
      float selectivity = float(satisfied_count) / size;
      if (selectivity <= 0.4) {
        // Supppose `size` is the minimal number of elements to ensure connectivity.
        auto estimatedFullTwoHopDistanceComp = (size * satisfied_count + satisfied_count) * 0.4;
        auto estimatedDirectedDistanceComp = size + (size - satisfied_count);
        int remaining = 10000000;
        // if (selectivity >= 0.2) {  // directed two-hop
        if (estimatedFullTwoHopDistanceComp > estimatedDirectedDistanceComp) {
          while (other_onehop_neighbors.size() > 0) {
            tableint cand_nbr = *other_onehop_neighbors.begin();
            other_onehop_neighbors.erase(other_onehop_neighbors.begin());
            vl->mass[cand_nbr] = vl->curV;
#ifdef USE_SSE
            if (other_onehop_neighbors.size() > 0) {
              _mm_prefetch(this->getDataByInternalId(*other_onehop_neighbors.begin()), _MM_HINT_T0);
            }
#endif
            ncomp++;
            dist_t dist = this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
            added_onehop_neighbors.emplace(-dist, cand_nbr);
            if (top_candidates.size() < efs || dist < upper_bound) {
              top_candidates.emplace(dist, cand_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-dist, -(int64_t)cand_nbr);
            }
          }
          remaining = size;
        }
        // }
        // undirected two-hop
        while (added_onehop_neighbors.size() > 0 && remaining > 0) {
          tableint cand_nbr = added_onehop_neighbors.top().second;
          added_onehop_neighbors.pop();

          tableint *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;
#ifdef USE_SSE
          _mm_prefetch((char *)(vl->mass + *(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs))), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + 1)), _MM_HINT_T0);
#endif

          for (int j = 0; j < twohop_size; j++) {
            tableint twohop_nbr = twohop_nbrs[j];
#ifdef USE_SSE
            _mm_prefetch((char *)(vl->mass + *(twohop_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs + j + 1))), _MM_HINT_T0);
            _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + j + 1)), _MM_HINT_T0);
#endif
            if (vl->mass[twohop_nbr] == vl->curV) continue;
            checked_count++;
            if (is_id_allowed != nullptr && !(*is_id_allowed)(twohop_nbr)) continue;
            vl->mass[twohop_nbr] = vl->curV;
            ncomp++;
            remaining--;
            dist_t twohop_nbr_dist =
                this->fstdistfunc_(query_data, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);
            added_count++;
            result_set.emplace(-twohop_nbr_dist, twohop_nbr);

            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);

              top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-twohop_nbr_dist, -(int64_t)twohop_nbr);
            }
          }
        }

        while (other_onehop_neighbors.size() > 0) {
          // Means we are running full two-hop search.
          // So we don't restrict on the number, i.e. `remaining`.
          tableint cand_nbr = *other_onehop_neighbors.begin();
          other_onehop_neighbors.erase(other_onehop_neighbors.begin());

          tableint *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;
#ifdef USE_SSE
          _mm_prefetch((char *)(vl->mass + *(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs))), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + 1)), _MM_HINT_T0);
#endif

          for (int j = 0; j < twohop_size; j++) {
            tableint twohop_nbr = twohop_nbrs[j];
#ifdef USE_SSE
            _mm_prefetch((char *)(vl->mass + *(twohop_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs + j + 1))), _MM_HINT_T0);
            _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + j + 1)), _MM_HINT_T0);
#endif
            if (vl->mass[twohop_nbr] == vl->curV) continue;
            checked_count++;
            if (is_id_allowed != nullptr && !(*is_id_allowed)(twohop_nbr)) continue;
            vl->mass[twohop_nbr] = vl->curV;
            ncomp++;
            remaining--;
            dist_t twohop_nbr_dist =
                this->fstdistfunc_(query_data, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);
            added_count++;
            result_set.emplace(-twohop_nbr_dist, twohop_nbr);

            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);
              top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-twohop_nbr_dist, -(int64_t)twohop_nbr);
            }
          }
        }
      }

      // Assess remaining one-hop neighbors.
      while (added_onehop_neighbors.size()) {
        tableint cand_nbr = added_onehop_neighbors.top().second;
        float cand_nbr_dist = -added_onehop_neighbors.top().first;
        added_onehop_neighbors.pop();

        // if (top_candidates.size() < efs || cand_nbr_dist <= upper_bound) {
        candidate_set.emplace(-cand_nbr_dist, cand_nbr);
        // };
      }

      // while (other_onehop_neighbors.size()) {
      //   tableint cand_nbr = other_onehop_neighbors.top().second;
      //   float cand_nbr_dist = -other_onehop_neighbors.top().first;
      //   other_onehop_neighbors.pop();

      //   if (top_candidates.size() < efs || cand_nbr_dist <= upper_bound) {
      //     candidate_set.emplace(-cand_nbr_dist, cand_nbr);
      //   };
      // }
      total_added_count += added_count;
      total_checked_count += checked_count;
    }
    sel = total_checked_count == 0 ? 0 : float(total_added_count) / total_checked_count;
  }

  void IterativeReentrantSearchKnnNavix(
      const void *query_data,
      const size_t k,
      RangeQuery<float> *is_id_allowed,
      std::priority_queue<std::pair<dist_t, int64_t>> &recycled_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp,
      float &sel
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    int total_added_count = 0, total_checked_count = 0;
    while (!candidate_set.empty()) {
      int added_count = 0, checked_count = 0, satisfied_count = 0;
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch((char *)(is_id_allowed->prefetch(*(cand_nbrs))), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      std::priority_queue<std::pair<dist_t, tableint>> added_onehop_neighbors;
      std::set<tableint> other_onehop_neighbors;
      // It may occur that the visited count is 0, which is not expected.
      // Work around this first.

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(is_id_allowed->prefetch(*(cand_nbrs + i + 1))), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (is_id_allowed == nullptr || (*is_id_allowed)(cand_nbr)) {
          satisfied_count++;
        }
        if (vl->mass[cand_nbr] == vl->curV) continue;
        checked_count++;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) {
          other_onehop_neighbors.insert(cand_nbr);
          continue;
        };
        vl->mass[cand_nbr] = vl->curV;

        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);

        result_set.emplace(-cand_nbr_dist, cand_nbr);
        added_count++;
        added_onehop_neighbors.emplace(-cand_nbr_dist, cand_nbr);

        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          // No need to put on candidate_set, because it will be traversed later.
          // candidate_set.emplace(-cand_nbr_dist, cand_nbr);
          // #ifdef USE_SSE
          //           _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
          // #endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) {
            auto top = top_candidates.top();
            recycled_candidates.emplace(-top.first, top.second);
            top_candidates.pop();
          }
          upper_bound = top_candidates.top().first;
        } else {
          recycled_candidates.emplace(-cand_nbr_dist, -(int64_t)cand_nbr);
        }
      }

      // TODO: The ratio is a tuning point.
      // adaptive local
      // float selectivity = checked_count == 0 ? 0 : (float)added_count / checked_count;
      float selectivity = float(satisfied_count) / size;
      if (selectivity <= 0.4) {
        // Supppose `size` is the minimal number of elements to ensure connectivity.
        auto estimatedFullTwoHopDistanceComp = (size * satisfied_count + satisfied_count) * 0.4;
        auto estimatedDirectedDistanceComp = size + (size - satisfied_count);
        int remaining = 10000000;
        // if (selectivity >= 0.2) {  // directed two-hop
        if (estimatedFullTwoHopDistanceComp > estimatedDirectedDistanceComp) {
          while (other_onehop_neighbors.size() > 0) {
            tableint cand_nbr = *other_onehop_neighbors.begin();
            other_onehop_neighbors.erase(other_onehop_neighbors.begin());
            vl->mass[cand_nbr] = vl->curV;
#ifdef USE_SSE
            if (other_onehop_neighbors.size() > 0) {
              _mm_prefetch(this->getDataByInternalId(*other_onehop_neighbors.begin()), _MM_HINT_T0);
            }
#endif
            ncomp++;
            dist_t dist = this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
            added_onehop_neighbors.emplace(-dist, cand_nbr);
            if (top_candidates.size() < efs || dist < upper_bound) {
              top_candidates.emplace(dist, cand_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-dist, -(int64_t)cand_nbr);
            }
          }
          remaining = size;
        }
        // }
        // undirected two-hop
        while (added_onehop_neighbors.size() > 0 && remaining > 0) {
          tableint cand_nbr = added_onehop_neighbors.top().second;
          added_onehop_neighbors.pop();

          tableint *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;
#ifdef USE_SSE
          _mm_prefetch((char *)(vl->mass + *(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs))), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + 1)), _MM_HINT_T0);
#endif

          for (int j = 0; j < twohop_size; j++) {
            tableint twohop_nbr = twohop_nbrs[j];
#ifdef USE_SSE
            _mm_prefetch((char *)(vl->mass + *(twohop_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs + j + 1))), _MM_HINT_T0);
            _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + j + 1)), _MM_HINT_T0);
#endif
            if (vl->mass[twohop_nbr] == vl->curV) continue;
            checked_count++;
            if (is_id_allowed != nullptr && !(*is_id_allowed)(twohop_nbr)) continue;
            vl->mass[twohop_nbr] = vl->curV;
            ncomp++;
            remaining--;
            dist_t twohop_nbr_dist =
                this->fstdistfunc_(query_data, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);
            added_count++;
            result_set.emplace(-twohop_nbr_dist, twohop_nbr);

            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);

              top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-twohop_nbr_dist, -(int64_t)twohop_nbr);
            }
          }
        }

        while (other_onehop_neighbors.size() > 0) {
          // Means we are running full two-hop search.
          // So we don't restrict on the number, i.e. `remaining`.
          tableint cand_nbr = *other_onehop_neighbors.begin();
          other_onehop_neighbors.erase(other_onehop_neighbors.begin());

          tableint *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;
#ifdef USE_SSE
          _mm_prefetch((char *)(vl->mass + *(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs))), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs)), _MM_HINT_T0);
          _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + 1)), _MM_HINT_T0);
#endif

          for (int j = 0; j < twohop_size; j++) {
            tableint twohop_nbr = twohop_nbrs[j];
#ifdef USE_SSE
            _mm_prefetch((char *)(vl->mass + *(twohop_nbrs + j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(is_id_allowed->prefetch(*(twohop_nbrs + j + 1))), _MM_HINT_T0);
            _mm_prefetch(this->getDataByInternalId(*(twohop_nbrs + j + 1)), _MM_HINT_T0);
#endif
            if (vl->mass[twohop_nbr] == vl->curV) continue;
            checked_count++;
            if (is_id_allowed != nullptr && !(*is_id_allowed)(twohop_nbr)) continue;
            vl->mass[twohop_nbr] = vl->curV;
            ncomp++;
            remaining--;
            dist_t twohop_nbr_dist =
                this->fstdistfunc_(query_data, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);
            added_count++;
            result_set.emplace(-twohop_nbr_dist, twohop_nbr);

            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);
              top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
              if (top_candidates.size() > efs) {
                auto top = top_candidates.top();
                recycled_candidates.emplace(-top.first, top.second);
                top_candidates.pop();
              };
              upper_bound = top_candidates.top().first;
            } else {
              recycled_candidates.emplace(-twohop_nbr_dist, -(int64_t)twohop_nbr);
            }
          }
        }
      }

      // Assess remaining one-hop neighbors.
      while (added_onehop_neighbors.size()) {
        tableint cand_nbr = added_onehop_neighbors.top().second;
        float cand_nbr_dist = -added_onehop_neighbors.top().first;
        added_onehop_neighbors.pop();

        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) {
          continue;
        }
        // if (top_candidates.size() < efs || cand_nbr_dist <= upper_bound) {
        candidate_set.emplace(-cand_nbr_dist, cand_nbr);
        // };
      }

      // while (other_onehop_neighbors.size()) {
      //   tableint cand_nbr = other_onehop_neighbors.top().second;
      //   float cand_nbr_dist = -other_onehop_neighbors.top().first;
      //   other_onehop_neighbors.pop();

      //   if (top_candidates.size() < efs || cand_nbr_dist <= upper_bound) {
      //     candidate_set.emplace(-cand_nbr_dist, cand_nbr);
      //   };
      // }
      total_added_count += added_count;
      total_checked_count += checked_count;
    }
    sel = total_checked_count == 0 ? 0 : float(total_added_count) / total_checked_count;
  }

  void IterativeReentrantSearchKnn(
      const void *query_data,
      const size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      AVL::Tree<std::pair<dist_t, labeltype>> &otree,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp,
      Out &out
  ) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t efs = std::max(k, this->ef_);
    float upper_bound = std::numeric_limits<dist_t>::max();
    {
      auto otree_start = std::chrono::high_resolution_clock::now();
      if (otree.size() != 0) {
        int idx = otree.size() < efs ? otree.size() : efs;
        upper_bound = otree.getValueGivenIndex(idx)->el.first;
      }
      auto otree_stop = std::chrono::high_resolution_clock::now();
      out.btree_time += std::chrono::duration_cast<std::chrono::nanoseconds>(otree_stop - otree_start).count();
    }

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        result_set.emplace(-cand_nbr_dist, cand_nbr);
        candidate_set.emplace(-cand_nbr_dist, cand_nbr);

        // if (otree.size() < efs || cand_nbr_dist < upper_bound) {
        //   candidate_set.emplace(-cand_nbr_dist, cand_nbr);
        // }
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
        {
          auto otree_start = std::chrono::high_resolution_clock::now();
          otree.insert(std::make_pair(cand_nbr_dist, cand_nbr));
          if (otree.size() >= efs) {
            upper_bound = otree.getValueGivenIndex(efs)->el.first;
          } else {
            upper_bound = otree.getValueGivenIndex(otree.size())->el.first;
          }
          auto otree_stop = std::chrono::high_resolution_clock::now();
          out.btree_time += std::chrono::duration_cast<std::chrono::nanoseconds>(otree_stop - otree_start).count();
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    out.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  void IterativeReentrantSearchKnn(
      const void *query_data,
      const size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      fc::BTreeMap<dist_t, std::pair<dist_t, labeltype>, 16> &btree,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      VisitedList *vl,
      int &ncomp,
      Out &out
  ) {
    auto start = std::chrono::high_resolution_clock::now();
    size_t efs = std::max(k, this->ef_);
    float upper_bound = std::numeric_limits<dist_t>::max();
    {
      auto btree_start = std::chrono::high_resolution_clock::now();
      if (btree.size() != 0) {
        int idx = btree.size() < efs ? btree.size() : efs;
        upper_bound = btree.kth(idx - 1).first;
      }
      auto btree_stop = std::chrono::high_resolution_clock::now();
      out.btree_time += std::chrono::duration_cast<std::chrono::nanoseconds>(btree_stop - btree_start).count();
    }

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch((char *)(vl->mass + *(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        result_set.emplace(-cand_nbr_dist, cand_nbr);
        candidate_set.emplace(-cand_nbr_dist, cand_nbr);

        // if (btree.size() < efs || cand_nbr_dist < upper_bound) {
        //   candidate_set.emplace(-cand_nbr_dist, cand_nbr);
        // }
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
        {
          auto btree_start = std::chrono::high_resolution_clock::now();
          btree[cand_nbr_dist] = std::make_pair(cand_nbr_dist, cand_nbr);
          if (btree.size() >= efs) {
            upper_bound = btree.kth(efs - 1).first;
          } else {
            upper_bound = btree.kth(btree.size() - 1).first;
          }
          auto btree_stop = std::chrono::high_resolution_clock::now();
          out.btree_time += std::chrono::duration_cast<std::chrono::nanoseconds>(btree_stop - btree_start).count();
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    out.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  void ReentrantSearchKnn(
      const void *query_data,
      size_t k,
      int nhops,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::vector<bool> &visited,
      BaseFilterFunctor *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound && top_candidates.size() >= efs) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (visited[cand_nbr]) continue;
        visited[cand_nbr] = true;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) continue;
        ncomp++;
        is_graph_ppsl[cand_nbr] = true;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) top_candidates.pop();
          upper_bound = top_candidates.top().first;
        }
      }
    }
  }

  void ReentrantSearchKnn(
      const void *query_data,
      size_t k,
      int nhops,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      VisitedList *vl,
      BaseFilterFunctor *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) continue;
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          is_graph_ppsl[cand_nbr] = true;
          if (top_candidates.size() > efs) top_candidates.pop();
          upper_bound = top_candidates.top().first;
        }
      }
    }
  }

  void ReentrantSearchKnnV2(
      const void *query_data,
      size_t k,
      int nhops,
      std::priority_queue<std::pair<dist_t, int64_t>> &top_candidates,
      std::priority_queue<std::pair<dist_t, int64_t>> &candidate_set,
      std::priority_queue<std::pair<dist_t, int64_t>> &frontiers,
      std::vector<bool> &visited,
      BaseFilterFunctor *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl,
      bool initial = false
  ) {
    if (initial) {
      tableint currObj = this->enterpoint_node_;
      dist_t curdist =
          this->fstdistfunc_(query_data, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);

      for (int level = this->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int *data;

          data = (unsigned int *)this->get_linklist(currObj, level);
          int size = this->getListCount(data);
          ncomp += size;

          tableint *datal = (tableint *)(data + 1);
          for (int i = 0; i < size; i++) {
            tableint cand = datal[i];
            if (cand < 0 || cand > this->max_elements_) throw std::runtime_error("cand error");
            dist_t d = this->fstdistfunc_(query_data, this->getDataByInternalId(cand), this->dist_func_param_);
            ncomp++;

            if (d < curdist) {
              curdist = d;
              currObj = cand;
              changed = true;
            }
          }
        }
      }

      candidate_set.emplace(-curdist, currObj);
    }

    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

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
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
      bool is_frontier = true;
      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
        if (visited[cand_nbr]) continue;
        visited[cand_nbr] = true;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(this->getExternalLabel(cand_nbr))) continue;
        is_frontier = false;
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
      if (is_frontier) {
        frontiers.emplace(-curr_dist, curr_obj);
      }
    }
  }

  //   void ReentrantSearchKnnBounded(
  //       const void *query_data,
  //       size_t k,
  //       float stop_bound,
  //       std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
  //       std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
  //       std::vector<bool> &visited,
  //       BaseFilterFunctor *is_id_allowed,
  //       int &ncomp,
  //       std::vector<bool> &is_graph_ppsl
  //   ) {
  //     size_t efs = std::max(k, this->ef_);
  //     auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

  //     while (!candidate_set.empty()) {
  //       auto curr_obj = candidate_set.top().second;
  //       auto curr_dist = -candidate_set.top().first;
  //       candidate_set.pop();

  //       if (curr_dist > upper_bound) {
  //         break;
  //       }

  //       unsigned int *cand_info = this->get_linklist0(curr_obj);
  //       int size = this->getListCount(cand_info);
  //       tableint *cand_nbrs = (tableint *)(cand_info + 1);
  // #ifdef USE_SSE
  //       _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
  //       _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
  // #endif

  //       for (int i = 0; i < size; i++) {
  //         tableint cand_nbr = cand_nbrs[i];
  // #ifdef USE_SSE
  //         _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
  // #endif
  //         if (visited[cand_nbr]) continue;
  //         visited[cand_nbr] = true;
  //         if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) continue;
  //         ncomp++;
  //         is_graph_ppsl[cand_nbr] = true;
  //         dist_t cand_nbr_dist =
  //             this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
  //         if (cand_nbr_dist < upper_bound) {
  //           candidate_set.emplace(-cand_nbr_dist, cand_nbr);
  // #ifdef USE_SSE
  //           _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
  // #endif
  //           top_candidates.emplace(cand_nbr_dist, cand_nbr);
  //           if (top_candidates.size() > efs) top_candidates.pop();
  //           upper_bound = top_candidates.top().first;
  //         }
  //       }
  //     }
  //   }

  void ReentrantSearchKnnBounded(
      const void *query_data,
      size_t k,
      float stop_bound,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      VisitedList *vl,
      RangeQuery<float> *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch(is_id_allowed->prefetch(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(is_id_allowed->prefetch(*(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch((vl->mass + *(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch((vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(is_id_allowed->prefetch(*(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) continue;
        ncomp++;
        is_graph_ppsl[cand_nbr] = true;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        if (cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) top_candidates.pop();
          upper_bound = top_candidates.top().first;
        }
      }
    }
  }

  void ReentrantSearchKnnBounded(
      const void *query_data,
      size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_graph,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_ivf,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      VisitedList *vl,
      RangeQuery<float> *is_id_allowed,
      int &ncomp,
      std::vector<bool> &is_graph_ppsl
  ) {
    size_t efs = std::max(k, this->ef_);
    auto upper_bound = top_graph.empty() ? std::numeric_limits<dist_t>::max() : top_graph.top().first;

    while (!candidate_set.empty()) {
      auto curr_obj = candidate_set.top().second;
      auto curr_dist = -candidate_set.top().first;
      candidate_set.pop();

      if (curr_dist > upper_bound || curr_dist > -top_ivf.top().first) {
        break;
      }

      unsigned int *cand_info = this->get_linklist0(curr_obj);
      int size = this->getListCount(cand_info);
      tableint *cand_nbrs = (tableint *)(cand_info + 1);
#ifdef USE_SSE
      _mm_prefetch(is_id_allowed->prefetch(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(is_id_allowed->prefetch(*(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch((vl->mass + *(cand_nbrs + 1)), _MM_HINT_T0);
      _mm_prefetch((vl->mass + *(cand_nbrs)), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*cand_nbrs), _MM_HINT_T0);
      _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + 1)), _MM_HINT_T0);
#endif

      for (int i = 0; i < size; i++) {
        tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
        _mm_prefetch((char *)(vl->mass + *(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(is_id_allowed->prefetch(*(cand_nbrs + i + 1)), _MM_HINT_T0);
        _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
        if (vl->mass[cand_nbr] == vl->curV) continue;
        vl->mass[cand_nbr] = vl->curV;
        if (is_id_allowed != nullptr && !(*is_id_allowed)(cand_nbr)) continue;
        ncomp++;
        is_graph_ppsl[cand_nbr] = true;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        if (cand_nbr_dist < upper_bound || top_graph.size() < efs) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
          top_graph.emplace(cand_nbr_dist, cand_nbr);
          if (top_graph.size() > efs) top_graph.pop();
          upper_bound = top_graph.top().first;
        }
      }
    }
  }
};