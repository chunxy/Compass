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

 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  // vector<vector<attr_t>> attrs_;
  attr_t* attrs_;
  rtree rtree_;

 public:
  CompassGraph(size_t d, size_t M, size_t efc, size_t max_elements)
      : space_(d), hnsw_(&space_, max_elements, M, efc), attrs_(new attr_t[max_elements * 2]), rtree_() {}

  int AddGraphPoint(const void *data_point, labeltype label) {
    hnsw_.addPoint(data_point, label, -1);
    return 1;
  }

  void AddAttrs(size_t n, const vector<vector<attr_t>> &attrs, labeltype *labels) {
    for (int i = 0; i < n; i++) {
      attrs_[i * 2] = attrs[i][0];
      attrs_[i * 2 + 1] = attrs[i][1];
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
      const int nrel,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(efs, k);
    hnsw_.setEf(efs_);
    vector<vector<pair<dist_t, labeltype>>> result(nq, vector<pair<dist_t, labeltype>>(k));
    size_t d = *(size_t *)space_.get_dist_func_param();

    VisitedList *vl = hnsw_.visited_list_pool_->getFreeVisitedList();

    RangeQuery<float> pred(l_bounds.data(), u_bounds.data(), attrs_, hnsw_.max_elements_, 2);
    point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
    box b(min_corner, max_corner);

    for (int q = 0; q < nq; q++) {
      auto rel_beg = rtree_.qbegin(geo::index::covered_by(b));
      auto rel_end = rtree_.qend();
      auto curr = rel_beg;

      std::priority_queue<pair<attr_t, int64_t>> top_candidates;
      std::priority_queue<pair<attr_t, int64_t>> candidate_set;

      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;

      {
        tableint currObj = hnsw_.enterpoint_node_;
        dist_t currDist = hnsw_.fstdistfunc_(
            (float *)query + q * d, hnsw_.getDataByInternalId(hnsw_.enterpoint_node_), hnsw_.dist_func_param_
        );

        for (int level = hnsw_.maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)hnsw_.get_linklist(currObj, level);
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

      while (curr != rel_end && top_candidates.size() < efs_) {
        int i = 0;
        while (i < nrel) {
          if (curr == rel_end) break;
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