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
  attr_t l_bound_, r_bound_;
  const vector<attr_t> *attrs_;  // index should be the labeltype

 public:
  RangeQuery(attr_t l_bound, attr_t u_bound, const vector<attr_t> *attrs)
      : l_bound_(l_bound), r_bound_(u_bound), attrs_(attrs) {}
  bool operator()(hnswlib::labeltype label) {
    if (label < attrs_->size()) {
      return l_bound_ <= (*attrs_)[label] && (*attrs_)[label] <= r_bound_;
    } else {
      return false;
    }
  }
};

template <typename attr_t>
class WindowQuery : public hnswlib::BaseFilterFunctor {
 private:
  vector<attr_t> l_bounds_, u_bounds_;
  const vector<vector<attr_t>> *attrs_;  // index should be the labeltype

 public:
  WindowQuery(vector<attr_t> l_bounds, vector<attr_t> u_bounds, const vector<vector<attr_t>> *attrs)
      : l_bounds_(l_bounds), u_bounds_(u_bounds), attrs_(attrs) {}
  bool operator()(hnswlib::labeltype label) {
    if (label < attrs_->size()) {
      for (int i = 0; i < l_bounds_.size(); i++) {
        if (l_bounds_[i] >= (*attrs_)[label][i] || (*attrs_)[label][i] >= u_bounds_[i]) {
          return false;
        }
      }
      return true;
    } else {
      return false;
    }
  }
};

template <typename attr_t, typename dist_t>
class RangedQueryStopCondition : public hnswlib::BaseSearchStopCondition<dist_t> {
 private:
  attr_t l_bound_, r_bound_;
  vector<attr_t> *attrs_;
  size_t curr_num_items_;
  vector<std::pair<dist_t, labeltype>> temp_result_;
  size_t window_size_;

 public:
  RangedQueryStopCondition(attr_t l_bound, attr_t u_bound, const vector<attr_t> *attrs)
      : l_bound_(l_bound), r_bound_(u_bound), attrs_(attrs) {
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
      if (l_bound_ <= attr_value && attr_value <= r_bound_) {
        stop_search = false;
        break;
      }
    }
    return stop_search;
  }
};