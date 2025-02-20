#pragma once

#include <fmt/core.h>
#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/filesystem.hpp>
#include <cassert>
#include <cstddef>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "../btree/btree_map.h"
#include "../hnswlib/hnswlib.h"
#include "../utils/predicate.h"
// #include "faiss/MetricType.h"
#include "methods/Pod.h"
#include "methods/ReentrantHNSW.h"
#include "space_l2.h"

namespace fs = boost::filesystem;
using coroutine_t = boost::coroutines2::coroutine<int>;
using std::pair;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassGraph1d {
  // using Parent = HierarchicalNSW<dist_t>;
  using candidate_queue = std::priority_queue<
      pair<dist_t, tableint>,
      vector<pair<dist_t, tableint>>,
      typename HierarchicalNSW<dist_t>::CompareByFirst>;

 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  vector<attr_t> attrs_;
  btree::btree_map<attr_t, tableint> btree_;
  size_t nrel_;

 public:
  CompassGraph1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nrel)
      : space_(d),
        hnsw_(&space_, max_elements, M, efc),
        attrs_(max_elements, std::numeric_limits<attr_t>::max()),
        btree_(),
        nrel_(nrel) {}

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
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(efs, k);
    hnsw_.setEf(efs_);
    vector<vector<pair<dist_t, labeltype>>> result(nq, vector<pair<dist_t, labeltype>>(k));
    size_t d = *(size_t *)space_.get_dist_func_param();

    for (int q = 0; q < nq; q++) {
      auto rel_beg = btree_.lower_bound(l_bound);
      auto rel_end = btree_.upper_bound(u_bound);

      std::priority_queue<pair<attr_t, int64_t>> top_candidates;
      std::priority_queue<pair<attr_t, int64_t>> candidate_set;

      vector<bool> visited(hnsw_.cur_element_count, false);
      auto tree_cand_push = [&, q](coroutine_t::push_type &push) {
        push(0);
        int cur_rel_cnt = 0;
        size_t efs = std::max((size_t)k, hnsw_.ef_);
        while (rel_beg != rel_end) {
          // auto label = (*rel_beg).second;
          // auto tableid = hnsw_.label_lookup_[label];
          tableint tableid = (*rel_beg).second;
          rel_beg++;
          if (visited[tableid]) continue;
          visited[tableid] = true;

          auto vect = hnsw_.getDataByInternalId(tableid);
          auto dist = hnsw_.fstdistfunc_((dist_t *)query + q * d, vect, space_.get_dist_func_param());
          metrics[q].ncomp++;
          candidate_set.emplace(-dist, tableid);
          top_candidates.emplace(dist, tableid);
          metrics[q].is_ivf_ppsl[tableid] = true;
          // metrics[q].is_ivf_ppsl[label] = true;
          if (top_candidates.size() > efs_) top_candidates.pop();
          cur_rel_cnt++;
          if (cur_rel_cnt == nrel_) {
            cur_rel_cnt = 0;
            push(nrel_);
          }
        }

        push(cur_rel_cnt);
      };

      coroutine_t::pull_type tree_cand_pull(tree_cand_push);

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      size_t total_proposed = 0;
      size_t max_dist_comp = 10;

      int max_try = 20;
      while (tree_cand_pull && (max_try-- > 0 || top_candidates.size() < efs_)) {
        tree_cand_pull();
        total_proposed += tree_cand_pull.get();
        hnsw_.ReentrantSearchKnn(
            (float *)query + q * d,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
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
    }


    return result;
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }

  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }
};