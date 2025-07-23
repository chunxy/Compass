#pragma once

#include <cstddef>
#include <utility>
#include <vector>
#include "../hnswlib/hnswlib.h"

using namespace hnswlib;
using std::vector;

template <typename attr_t>
class RangeQuery : public hnswlib::BaseFilterFunctor {
 private:
  const attr_t *l_bound_, *u_bound_;  // of dimension d_
  const attr_t *attrs_;               // index should be the labeltype
  size_t n_, d_;

 public:
  RangeQuery(const attr_t *l_bound, const attr_t *u_bound, const attr_t *attrs, size_t n, size_t d)
      : l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs), n_(n), d_(d) {}
  bool operator()(hnswlib::labeltype label) override {
    if (label < n_) {
      for (int i = 0; i < d_; i++) {
        if (l_bound_[i] > attrs_[label * d_ + i] || attrs_[label * d_ + i] > u_bound_[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
};

template <typename attr_t>
class WindowQuery : public hnswlib::BaseFilterFunctor {
 private:
  const attr_t *l_bound_, *u_bound_;  // of dimension d
  const attr_t *attrs_;               // index should be the labeltype
  size_t n_, d_;

 public:
  WindowQuery(const attr_t *l_bound, const attr_t *u_bound, const attr_t *attrs, size_t n, size_t d)
      : l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs), n_(n), d_(d) {}
  bool operator()(hnswlib::labeltype label) {
    if (label < n_) {
      for (int i = 0; i < d_; i++) {
        if (l_bound_[i] >= attrs_[label * d_ + i] || attrs_[label * d_ + i] >= u_bound_[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }
};

template <typename attr_t, typename dist_t>
class RangedQueryStopCondition : public hnswlib::BaseSearchStopCondition<dist_t> {
 private:
  attr_t l_bound_, u_bound_;
  vector<attr_t> *attrs_;
  size_t curr_num_items_;
  vector<std::pair<dist_t, labeltype>> temp_result_;
  size_t window_size_;

 public:
  RangedQueryStopCondition(attr_t l_bound, attr_t u_bound, const vector<attr_t> *attrs)
      : l_bound_(l_bound), u_bound_(u_bound), attrs_(attrs) {
    curr_num_items_ = 0;
  }

  void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) override {
    curr_num_items_ += 1;
    temp_result_.emplace(dist, label);
  }

  bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) override {
    bool stop_search = true;
    size_t i = 0;
    for (auto it = temp_result_.rbegin(); it != temp_result_.rend() && i < window_size_; it++, i++) {
      auto attr_value = attrs_[it.second];
      if (l_bound_ <= attr_value && attr_value <= u_bound_) {
        stop_search = false;
        break;
      }
    }
    return stop_search;
  }
};