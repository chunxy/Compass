#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <vector>
#include "ReentrantHNSW.h"

using std::pair;
using std::priority_queue;
using std::string;
using std::vector;

template <typename dist_t>
class IterativeSearch;

template <typename dist_t>
class IterativeSearchState {
  friend class IterativeSearch<dist_t>;
  const dist_t *query_;
  const int k_;
  priority_queue<pair<dist_t, labeltype>> all_candidates_;
  priority_queue<pair<dist_t, labeltype>> top_candidates_;
  priority_queue<pair<dist_t, labeltype>> candidate_set_;
  priority_queue<pair<dist_t, labeltype>> result_set_;
  priority_queue<pair<dist_t, labeltype>> batch_rz_;
  vector<bool> visited_;
  int ncomp_;
  int total_;

  IterativeSearchState(const dist_t *query, int k) : query_(query), k_(k) {}
};

template <typename dist_t>
class IterativeSearch {
 private:
  int n_, d_, M_, efc_;
  int batch_k_, initial_efs_, delta_efs_;  // a decent combo of k and efs for search
  ReentrantHNSW<dist_t> *hnsw_;

  int UpdateNext(IterativeSearchState<dist_t> *state) {
    state->top_candidates_ = state->all_candidates_;
    while (state->top_candidates_.size() > hnsw_->ef_) {
      state->top_candidates_.pop();
    }
    hnsw_->ReentrantSearchKnn(
        state->query_,
        this->batch_k_,
        state->all_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->visited_,
        state->ncomp_
    );
    int cnt = 0;
    while (cnt < this->batch_k_ && !state->result_set_.empty()) {
      auto top = state->result_set_.top();
      state->result_set_.pop();
      state->batch_rz_.emplace(top.first, top.second);
      cnt++;
    }
    hnsw_->setEf(hnsw_->ef_ + this->delta_efs_);  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  bool HasNext(IterativeSearchState<dist_t> *state) { return !state->batch_rz_.empty(); }

 public:
  IterativeSearch(int n, int d, int M, int efc, const string &path, int batch_k = 10, int delta_efs = 50)
      : n_(n),
        d_(d),
        M_(M),
        efc_(efc),
        batch_k_(batch_k),
        delta_efs_(delta_efs),
        hnsw_(new ReentrantHNSW<dist_t>(new L2Space(d), path, false, n)) {
    hnsw_->setEf(std::max(batch_k, delta_efs));
    this->initial_efs_ = hnsw_->ef_;
  }

  IterativeSearchState<dist_t> *Open(const dist_t *query, int k) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> *state = new IterativeSearchState<dist_t>(query, k);
    state->visited_.resize(n_, false);
    state->ncomp_ = 0;
    state->total_ = 0;
    {
      tableint curr_obj = this->hnsw_->enterpoint_node_;
      dist_t curr_dist =
          this->hnsw_->fstdistfunc_(query, this->hnsw_->getDataByInternalId(curr_obj), this->hnsw_->dist_func_param_);

      for (int level = this->hnsw_->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
          changed = false;
          unsigned int *data;

          data = (unsigned int *)this->hnsw_->get_linklist(curr_obj, level);
          int size = this->hnsw_->getListCount(data);

          tableint *datal = (tableint *)(data + 1);
          for (int i = 0; i < size; i++) {
            tableint cand = datal[i];

            if (cand < 0 || cand > this->hnsw_->max_elements_) throw std::runtime_error("cand error");
            dist_t d =
                this->hnsw_->fstdistfunc_(query, this->hnsw_->getDataByInternalId(cand), this->hnsw_->dist_func_param_);
            state->ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state->visited_[curr_obj] = true;
      state->candidate_set_.emplace(-curr_dist, curr_obj);
      state->all_candidates_.emplace(curr_dist, curr_obj);

      // hnsw_->ReentrantSearchKnn(
      //     state->query_,
      //     this->k_,
      //     state->top_candidates_,
      //     state->cur_candidates_,
      //     state->candidate_set_,
      //     state->result_set_,
      //     state->visited_,
      //     state->ncomp_
      // );
      UpdateNext(state);
    }
    return state;
  }

  pair<dist_t, labeltype> Next(IterativeSearchState<dist_t> *state) {
    if (state->total_ >= state->k_) {
      return {-1, -1};
    }
    if (HasNext(state)) {
      auto top = state->batch_rz_.top();
      state->batch_rz_.pop();
      state->total_++;
      return {-top.first, top.second};
    } else {
      int cnt = UpdateNext(state);
      if (cnt == 0) {
        return {-1, -1};
      }
    }
    return Next(state);
  }

  int GetNcomp(IterativeSearchState<dist_t> *state) { return state->ncomp_; }

  void Close(IterativeSearchState<dist_t> *state) {
    delete state;
    hnsw_->setEf(this->initial_efs_);
  }
};