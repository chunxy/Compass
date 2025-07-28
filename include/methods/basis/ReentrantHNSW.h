#pragma once

#include <algorithm>
#include <cstddef>
#include "avl.h"
#include "hnswlib/hnswlib.h"

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

  void ReentrantSearchKnn(
      const void *query_data,
      const size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> &recycled_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &top_candidates,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      std::vector<bool> &visited,
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
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);

        result_set.emplace(-cand_nbr_dist, cand_nbr);
        candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
        if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
          top_candidates.emplace(cand_nbr_dist, cand_nbr);
          if (top_candidates.size() > efs) {
            auto top = top_candidates.top();
            recycled_candidates.emplace(-top.first, top.second);
            top_candidates.pop();
          }
          upper_bound = top_candidates.top().first;
        } else {
          recycled_candidates.emplace(-cand_nbr_dist, cand_nbr);
        }
      }
    }
  }

  void ReentrantSearchKnn(
      const void *query_data,
      const size_t k,
      std::priority_queue<std::pair<dist_t, labeltype>> &candidate_set,
      AVL::Tree<std::pair<dist_t, labeltype>> &otree,
      std::priority_queue<std::pair<dist_t, labeltype>> &result_set,
      std::vector<bool> &visited,
      int &ncomp
  ) {
    size_t efs = std::max(k, this->ef_);
    float upper_bound = std::numeric_limits<dist_t>::max();
    if (otree.size() != 0) {
      int idx = otree.size() < efs ? otree.size() : efs;
      upper_bound = otree.getValueGivenIndex(idx)->el.first;
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
        ncomp++;
        dist_t cand_nbr_dist =
            this->fstdistfunc_(query_data, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
        result_set.emplace(-cand_nbr_dist, cand_nbr);

        if (otree.size() < efs || cand_nbr_dist < upper_bound) {
          candidate_set.emplace(-cand_nbr_dist, cand_nbr);
        }
#ifdef USE_SSE
        _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
        otree.insert(std::make_pair(cand_nbr_dist, cand_nbr));
        if (otree.size() >= efs) {
          upper_bound = otree.getValueGivenIndex(efs)->el.first;
        } else {
          upper_bound = otree.getValueGivenIndex(otree.size())->el.first;
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

  void ReentrantSearchKnnBounded(
      const void *query_data,
      size_t k,
      float stop_bound,
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
        if (visited[cand_nbr]) continue;
        visited[cand_nbr] = true;
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
      float stop_bound,
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
};