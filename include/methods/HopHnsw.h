#pragma once

#include <algorithm>
#include <boost/filesystem.hpp>
#include <cstddef>
#include <utility>
#include <vector>
#include "../hnswlib/hnswlib.h"
#include "utils/Pod.h"
#include "visited_list_pool.h"

using namespace hnswlib;
using std::pair;
using std::vector;

namespace fs = boost::filesystem;

template <typename dist_t>
class HopHnsw : public HierarchicalNSW<dist_t> {
 public:
  HopHnsw(
      SpaceInterface<dist_t> *s,
      size_t max_elements,
      size_t M = 16,
      size_t efc = 200,
      size_t random_seed = 100,
      bool allow_replace_deleted = false
  )
      : HierarchicalNSW<dist_t>(s, max_elements, M, efc, random_seed, allow_replace_deleted) {}

  HopHnsw(
      SpaceInterface<dist_t> *s,
      const std::string &location,
      bool nmslib = false,
      size_t max_elements = 0,
      bool allow_replace_deleted = false
  )
      : HierarchicalNSW<dist_t>(s, location, nmslib, max_elements, allow_replace_deleted) {}

  template <typename attr_t>
  vector<vector<pair<dist_t, labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const int efs,
      BaseFilterFunctor *is_id_allowed,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(efs, k);
    this->setEf(efs_);
    vector<vector<pair<dist_t, labeltype>>> result(nq, vector<pair<dist_t, labeltype>>(k));
    size_t d = *(size_t *)(this->dist_func_param_);

    VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
    for (int q = 0; q < nq; q++) {
      std::priority_queue<pair<attr_t, int64_t>> top_candidates;
      std::priority_queue<pair<attr_t, int64_t>> candidate_set;

      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;

      {
        tableint currObj = this->enterpoint_node_;
        dist_t currDist = this->fstdistfunc_(
            (float *)query + q * d, this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_
        );

        for (int level = this->maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (tableint *)this->get_linklist(currObj, level);
            int size = this->getListCount(data);
            metrics[q].ncomp += size;

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > this->max_elements_) throw std::runtime_error("cand error");
              dist_t dist =
                  this->fstdistfunc_((float *)query + q * d, this->getDataByInternalId(cand), this->dist_func_param_);

              if (dist < currDist) {
                currDist = dist;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
        visited[currObj] = visited_tag;
        candidate_set.emplace(-currDist, currObj);
        if ((*is_id_allowed)(currObj)) top_candidates.emplace(currDist, currObj);
      }

      auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;

      while (true) {
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

        int precision_comp = this->maxM0_;
        for (int i = 0; i < size; i++) {
          tableint cand_nbr = cand_nbrs[i];
#ifdef USE_SSE
          _mm_prefetch(this->getDataByInternalId(*(cand_nbrs + i + 1)), _MM_HINT_T0);
#endif
          if (visited[cand_nbr] == visited_tag) continue;
          visited[cand_nbr] = visited_tag;
          metrics[q].ncomp++;
          dist_t cand_nbr_dist = this->fstdistfunc_((float *)query + q * d, this->getDataByInternalId(cand_nbr), this->dist_func_param_);
          if (top_candidates.size() < efs || cand_nbr_dist < upper_bound) {
            candidate_set.emplace(-cand_nbr_dist, cand_nbr);
#ifdef USE_SSE
            _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
            metrics[q].is_graph_ppsl[cand_nbr] = true;
            if (is_id_allowed == nullptr || (*is_id_allowed)(cand_nbr)) {
              top_candidates.emplace(cand_nbr_dist, cand_nbr);
              if (top_candidates.size() > efs) top_candidates.pop();
              upper_bound = top_candidates.top().first;
              precision_comp--;
            }
          }
        }

        for (int i = 0; i < size; i++) {
          if (precision_comp <= 0) break;

          tableint cand_nbr = cand_nbrs[i];
          unsigned int *twohop_info = this->get_linklist0(cand_nbr);
          int twohop_size = this->getListCount(twohop_info);
          tableint *twohop_nbrs = twohop_info + 1;

          for (int j = 0; j < twohop_size && precision_comp > 0; j++) {
            tableint twohop_nbr = twohop_nbrs[j];
            if (visited[twohop_nbr] == visited_tag) continue;
            visited[twohop_nbr] = visited_tag;
            metrics[q].ncomp++;

            dist_t twohop_nbr_dist =
                this->fstdistfunc_((float *)query + q * d, this->getDataByInternalId(twohop_nbr), this->dist_func_param_);
            if (top_candidates.size() < efs || twohop_nbr_dist < upper_bound) {
              candidate_set.emplace(-twohop_nbr_dist, twohop_nbr);
#ifdef USE_SSE
              _mm_prefetch(this->getDataByInternalId(candidate_set.top().second), _MM_HINT_T0);
#endif
              metrics[q].is_graph_ppsl[twohop_nbr] = true;
              if (is_id_allowed == nullptr || (*is_id_allowed)(twohop_nbr)) {
                top_candidates.emplace(twohop_nbr_dist, twohop_nbr);
                if (top_candidates.size() > efs) top_candidates.pop();
                upper_bound = top_candidates.top().first;
                precision_comp--;
              }
            }
          }
        }
      }

      while (top_candidates.size() > k) top_candidates.pop();
      auto sz = top_candidates.size();
      result[q].resize((sz));
      while (top_candidates.size() > 0) {
        result[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }

      vl->reset();
    }
    this->visited_list_pool_->releaseVisitedList(vl);

    return result;
  }

  void LoadGraph(fs::path path, SpaceInterface<dist_t> *space) { this->loadIndex(path.string(), space); }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->saveIndex(path.string());
  }
};