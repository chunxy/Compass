#pragma once

#include <fmt/core.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "basis/ReentrantHNSW.h"
#include "btree_map.h"
#include "hnswlib/hnswlib.h"
#include "utils/Pod.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
using std::pair;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassGraph1d {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  vector<attr_t> attrs_;
  btree::btree_map<attr_t, tableint> btree_;

 public:
  CompassGraph1d(size_t d, size_t M, size_t efc, size_t max_elements)
      : space_(d),
        hnsw_(&space_, max_elements, M, efc),
        attrs_(max_elements, std::numeric_limits<attr_t>::max()),
        btree_() {}

  int AddGraphPoint(const void *data_point, labeltype label) {
    hnsw_.addPoint(data_point, label, -1);
    return 1;
  }

  void AddAttrs(size_t n, attr_t *attrs, labeltype *labels) {
    for (int i = 0; i < n; i++) {
      attrs_[i] = attrs[i];
      btree_.insert({attrs[i], labels[i]});
    }
  }

  // Invoking this method implies that we're already at the base layer.
  vector<vector<pair<dist_t, labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t l_bound,
      const attr_t u_bound,
      const int efs,
      const int nrel,
      vector<QueryMetric> &metrics
  ) {
    auto efs_ = std::max(efs, k);
    hnsw_.setEf(efs_);
    vector<vector<pair<dist_t, labeltype>>> result(nq, vector<pair<dist_t, labeltype>>(k));
    size_t d = *(size_t *)space_.get_dist_func_param();

    VisitedList *vl = hnsw_.visited_list_pool_->getFreeVisitedList();

    for (int q = 0; q < nq; q++) {
      auto pred_beg = btree_.lower_bound(l_bound);
      auto pred_end = btree_.upper_bound(u_bound);
      auto curr = pred_beg;

      std::priority_queue<pair<dist_t, labeltype>> top_candidates;
      std::priority_queue<pair<dist_t, labeltype>> candidate_set;

      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;

      RangeQuery<float> pred(&l_bound, &u_bound, attrs_.data(), hnsw_.max_elements_, 1);

      {
        tableint currObj = hnsw_.enterpoint_node_;
        dist_t currDist = hnsw_.fstdistfunc_(
            (float *)query + q * d, hnsw_.getDataByInternalId(hnsw_.enterpoint_node_), hnsw_.dist_func_param_
        );

        for (int level = hnsw_.maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            tableint *data;

            data = (tableint *)hnsw_.get_linklist(currObj, level);
            int size = hnsw_.getListCount(data);
            metrics[q].ncomp += size;

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > hnsw_.max_elements_) throw std::runtime_error("cand error");
              dist_t dist =
                  hnsw_.fstdistfunc_((float *)query + q * d, hnsw_.getDataByInternalId(cand), hnsw_.dist_func_param_);

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
        metrics[q].is_graph_ppsl[currObj] = true;
        if (pred(currObj)) top_candidates.emplace(currDist, currObj);
      }

      while (curr != pred_end && top_candidates.size() < efs_) {
        int i = 0;
        while (i < nrel) {
          if (curr == pred_end) break;
          tableint id = (*curr).second;
          curr++;
          if (visited[id] == visited_tag) continue;
          visited[id] = visited_tag;
#ifdef USE_SSE
          _mm_prefetch(hnsw_.getDataByInternalId((*curr).second), _MM_HINT_T0);
#endif
          auto vect = hnsw_.getDataByInternalId(id);
          auto dist = hnsw_.fstdistfunc_(vect, (float *)(query) + q * d, space_.get_dist_func_param());
          metrics[q].ncomp++;
          metrics[q].is_ivf_ppsl[id] = true;
          candidate_set.emplace(-dist, id);
          top_candidates.emplace(dist, id);
          if (top_candidates.size() > efs_) top_candidates.pop();
          i++;
        }
        hnsw_.ReentrantSearchKnn(
            (float *)query + q * d,
            k,
            -1,
            top_candidates,
            candidate_set,
            vl,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
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

    hnsw_.visited_list_pool_->releaseVisitedList(vl);
    return result;
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }

  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }
};