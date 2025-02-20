#pragma once

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/index/predicates.hpp>
#include <cassert>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>
#include "../hnswlib/hnswlib.h"
#include "../utils/predicate.h"
// #include "faiss/MetricType.h"
#include "methods/Pod.h"
#include "methods/ReentrantHNSW.h"
#include "space_l2.h"

namespace fs = boost::filesystem;
namespace geo = boost::geometry;

using point = geo::model::point<float, 2, geo::cs::cartesian>;
using box = geo::model::box<point>;
using value = std::pair<point, labeltype>;
using rtree = geo::index::rtree<value, geo::index::quadratic<16>>;

using std::pair;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassGraph {
  // using Parent = HierarchicalNSW<dist_t>;
  using candidate_queue = std::priority_queue<
      pair<dist_t, tableint>,
      vector<pair<dist_t, tableint>>,
      typename HierarchicalNSW<dist_t>::CompareByFirst>;

 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  vector<vector<attr_t>> attrs_;
  rtree rtree_;
  size_t nrel_;

 public:
  CompassGraph(size_t d, size_t M, size_t efc, size_t max_elements, size_t nrel)
      : space_(d),
        hnsw_(&space_, max_elements, M, efc),
        attrs_(max_elements, vector<attr_t>()),
        rtree_(),
        nrel_(nrel) {}

  int AddGraphPoint(const void *data_point, labeltype label) {
    hnsw_.addPoint(data_point, label, -1);
    return 1;
  }

  void AddAttrs(size_t n, const vector<vector<attr_t>> &attrs, labeltype *labels) {
    for (int i = 0; i < n; i++) {
      attrs_[i] = attrs[i];
      point p(attrs[i][0], attrs[i][1]);
      rtree_.insert(std::make_pair(p, labels[i]));
    }
  }

  // Invoking this method implies that we're already at the base layer.
  vector<vector<pair<dist_t, labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const vector<attr_t> &l_bounds,
      const vector<attr_t> &u_bounds,
      const int efs,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(efs, k);
    hnsw_.setEf(efs_);
    vector<vector<pair<dist_t, labeltype>>> result(nq, vector<pair<dist_t, labeltype>>(k));
    size_t d = *(size_t *)space_.get_dist_func_param();

    for (int q = 0; q < nq; q++) {
      point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
      box b(min_corner, max_corner);
      auto rel_beg = rtree_.qbegin(geo::index::covered_by(b));
      auto rel_end = rtree_.qend();

      std::priority_queue<pair<attr_t, int64_t>> top_candidates;
      std::priority_queue<pair<attr_t, int64_t>> candidate_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      WindowQuery<float> pred(l_bounds, u_bounds, &attrs_);

      int max_try = 20;
      while ((max_try-- > 0 || top_candidates.size() < efs_)) {
        if (rel_beg != rel_end) {
          int cur_rel_cnt = 0;
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
              break;
            }
          }
        }
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