#pragma once

#include <algorithm>
#include <queue>
#include <string>
#include <vector>
#include "ReentrantHNSW.h"
#include "hnswlib/hnswlib.h"
#include "utils/out.h"
#include "utils/predicate.h"

using std::pair;
using std::priority_queue;
using std::string;
using std::vector;

template <typename dist_t>
class IterativeSearch;

template <typename dist_t>
class IterativeSearchState {
  friend class IterativeSearch<dist_t>;
  const void *query_;
  size_t k_;
  int batch_k_;
  int cur_cnt;
  bool has_ran_;

  priority_queue<pair<dist_t, int64_t>> recycled_candidates_;  // min heap
 public:
  priority_queue<pair<dist_t, labeltype>> top_candidates_;  // max heap
  priority_queue<pair<dist_t, labeltype>> candidate_set_;   // min heap
  priority_queue<pair<dist_t, labeltype>> result_set_;      // min heap
  float sel_;

 private:
  priority_queue<pair<dist_t, labeltype>> batch_rz_;              // min heap
  AVL::Tree<std::pair<dist_t, labeltype>> otree_;                 // top candidates alternative
  fc::BTreeMap<dist_t, std::pair<dist_t, labeltype>, 16> btree_;  // top candidates alternative
  VisitedList *vl_;
  int ncomp_;
  int total_;

 public:
  Out out_;
  IterativeSearchState(const void *query, size_t k)
      : query_(query), k_(k), ncomp_(0), total_(0), batch_k_(1), cur_cnt(0), has_ran_(false) {}
  IterativeSearchState(const void *query, size_t k, int batch_k)
      : query_(query), k_(k), ncomp_(0), total_(0), batch_k_(batch_k), cur_cnt(0), has_ran_(false) {}
};

template <typename dist_t>
class IterativeSearch {
 public:
  const int n_;
  int batch_k_, delta_efs_, initial_efs_;  // a decent combo of batch_k and delta_efs for search
  ReentrantHNSW<dist_t> *hnsw_;
  VisitedList *vl_;

  int UpdateNext(IterativeSearchState<dist_t> *state) {
#ifndef BENCH
    auto start = std::chrono::high_resolution_clock::now();
#endif
    while (!state->recycled_candidates_.empty() && state->top_candidates_.size() < hnsw_->ef_) {
      auto top = state->recycled_candidates_.top();
      if (top.second < 0) {
        state->top_candidates_.emplace(-top.first, -top.second);
        state->candidate_set_.emplace(top.first, -top.second);
        state->result_set_.emplace(top.first, -top.second);
      } else {
        state->top_candidates_.emplace(-top.first, top.second);
      }
      state->recycled_candidates_.pop();
    }
#ifndef BENCH
    auto end = std::chrono::high_resolution_clock::now();
    state->out_.pop_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
#ifndef BENCH
    start = std::chrono::high_resolution_clock::now();
#endif
    hnsw_->IterativeReentrantSearchKnn(
        state->query_,
        this->batch_k_,
        state->recycled_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->vl_,
        state->ncomp_
    );
#ifndef BENCH
    end = std::chrono::high_resolution_clock::now();
    state->out_.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
    int cnt = state->result_set_.size();
    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  int UpdateNextTwoHop(IterativeSearchState<dist_t> *state, RangeQuery<float> *pred) {
#ifndef BENCH
    auto start = std::chrono::high_resolution_clock::now();
#endif
    while (!state->recycled_candidates_.empty() && state->top_candidates_.size() < hnsw_->ef_) {
      auto top = state->recycled_candidates_.top();
      if (top.second < 0) {
        state->top_candidates_.emplace(-top.first, -top.second);
        state->candidate_set_.emplace(top.first, -top.second);
        if (pred == nullptr || (*pred)(-top.second)) {
          state->result_set_.emplace(top.first, -top.second);
        }
      } else {
        state->top_candidates_.emplace(-top.first, top.second);
      }
      state->recycled_candidates_.pop();
    }
#ifndef BENCH
    auto end = std::chrono::high_resolution_clock::now();
    state->out_.pop_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
#ifndef BENCH
    start = std::chrono::high_resolution_clock::now();
#endif
    hnsw_->IterativeReentrantSearchKnnTwoHop(
        state->query_,
        this->batch_k_,
        pred,
        state->recycled_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->vl_,
        state->ncomp_,
        state->sel_,
        &state->out_
    );
#ifndef BENCH
    end = std::chrono::high_resolution_clock::now();
    state->out_.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
    int cnt = state->result_set_.size();
    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  int UpdateNextTwoHop(IterativeSearchState<dist_t> *state, BaseFilterFunctor *bitset) {
#ifndef BENCH
    auto start = std::chrono::high_resolution_clock::now();
#endif
    while (!state->recycled_candidates_.empty() && state->top_candidates_.size() < hnsw_->ef_) {
      auto top = state->recycled_candidates_.top();
      if (top.second < 0) {
        state->top_candidates_.emplace(-top.first, -top.second);
        state->candidate_set_.emplace(top.first, -top.second);
        if (bitset == nullptr || (*bitset)(-top.second)) {
          state->result_set_.emplace(top.first, -top.second);
        }
      } else {
        state->top_candidates_.emplace(-top.first, top.second);
      }
      state->recycled_candidates_.pop();
    }
#ifndef BENCH
    auto end = std::chrono::high_resolution_clock::now();
    state->out_.pop_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
#ifndef BENCH
    start = std::chrono::high_resolution_clock::now();
#endif
    hnsw_->IterativeReentrantSearchKnnTwoHop(
        state->query_,
        this->batch_k_,
        bitset,
        state->recycled_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->vl_,
        state->ncomp_,
        state->sel_,
        &state->out_
    );
#ifndef BENCH
    end = std::chrono::high_resolution_clock::now();
    state->out_.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
    int cnt = state->result_set_.size();
    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  int UpdateNextNavix(IterativeSearchState<dist_t> *state, RangeQuery<float> *pred) {
#ifndef BENCH
    auto start = std::chrono::high_resolution_clock::now();
#endif
    while (!state->recycled_candidates_.empty() && state->top_candidates_.size() < hnsw_->ef_) {
      auto top = state->recycled_candidates_.top();
      if (top.second < 0) {
        state->top_candidates_.emplace(-top.first, -top.second);
        state->candidate_set_.emplace(top.first, -top.second);
      } else {
        state->top_candidates_.emplace(-top.first, top.second);
      }
      state->recycled_candidates_.pop();
    }
#ifndef BENCH
    auto end = std::chrono::high_resolution_clock::now();
    state->out_.pop_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
#ifndef BENCH
    start = std::chrono::high_resolution_clock::now();
#endif
    hnsw_->IterativeReentrantSearchKnnNavix(
        state->query_,
        this->batch_k_,
        pred,
        state->recycled_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->vl_,
        state->ncomp_,
        state->sel_
    );
#ifndef BENCH
    end = std::chrono::high_resolution_clock::now();
    state->out_.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
    int cnt = 0;
    while (cnt < this->batch_k_ && !state->result_set_.empty()) {
      auto top = state->result_set_.top();
      state->result_set_.pop();
      state->batch_rz_.emplace(top.first, top.second);
      cnt++;
    }
    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  int UpdateNextFiltered(IterativeSearchState<dist_t> *state, BaseFilterFunctor *pred) {
#ifndef BENCH
    auto start = std::chrono::high_resolution_clock::now();
#endif
    while (!state->recycled_candidates_.empty() && state->top_candidates_.size() < hnsw_->ef_) {
      auto top = state->recycled_candidates_.top();
      if (top.second < 0) {
        state->top_candidates_.emplace(-top.first, -top.second);
        state->candidate_set_.emplace(top.first, -top.second);
      } else {
        state->top_candidates_.emplace(-top.first, top.second);
      }
      state->recycled_candidates_.pop();
    }
#ifndef BENCH
    auto end = std::chrono::high_resolution_clock::now();
    state->out_.pop_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
#ifndef BENCH
    start = std::chrono::high_resolution_clock::now();
#endif
    hnsw_->IterativeReentrantSearchKnnPostFiltered(
        state->query_,
        this->batch_k_,
        pred,
        state->recycled_candidates_,
        state->top_candidates_,
        state->candidate_set_,
        state->result_set_,
        state->vl_,
        state->ncomp_
    );
#ifndef BENCH
    end = std::chrono::high_resolution_clock::now();
    state->out_.search_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#endif
    int cnt = 0;
    if (state->result_set_.size() < this->batch_k_) {
      std::swap(state->result_set_, state->batch_rz_);
      cnt = state->batch_rz_.size();
    } else {
      while (cnt < this->batch_k_ && !state->result_set_.empty()) {
        auto top = state->result_set_.top();
        state->result_set_.pop();
        state->batch_rz_.emplace(top.first, top.second);
        cnt++;
      }
    }

    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  int UpdateNextNeo(IterativeSearchState<dist_t> *state) {
    hnsw_->IterativeReentrantSearchKnn(
        state->query_,
        this->batch_k_,
        state->candidate_set_,
        state->btree_,
        state->result_set_,
        state->vl_,
        state->ncomp_,
        state->out_
    );
    // hnsw_->IterativeReentrantSearchKnn(
    //     state->query_,
    //     this->batch_k_,
    //     state->candidate_set_,
    //     state->otree_,
    //     state->result_set_,
    //     state->vl_,
    //     state->ncomp_,
    //     state->out_
    // );
    int cnt = 0;
    while (cnt < this->batch_k_ && !state->result_set_.empty()) {
      auto top = state->result_set_.top();
      state->result_set_.pop();
      state->batch_rz_.emplace(top.first, top.second);
      cnt++;
    }
    hnsw_->setEf(std::min(hnsw_->ef_ + this->delta_efs_, state->k_));  // expand the efs for next batch search
    // Remember to reset the ef of hnsw_ to the initial value when closing the state.
    return cnt;
  }

  bool HasNext(IterativeSearchState<dist_t> *state) { return !state->batch_rz_.empty(); }

 public:
  IterativeSearch(int n, int d, const string &path, SpaceInterface<dist_t> *s)
      : n_(n), batch_k_(10), delta_efs_(20), initial_efs_(20), hnsw_(new ReentrantHNSW<dist_t>(s, path, false, n)) {
    hnsw_->setEf(this->initial_efs_);
  }

  IterativeSearch(int n, int d, SpaceInterface<dist_t> *s, int M_cg)
      : n_(n), batch_k_(10), delta_efs_(20), initial_efs_(20), hnsw_(new ReentrantHNSW<dist_t>(s, n, M_cg, 200)) {
    hnsw_->setEf(this->initial_efs_);
  }

  // VistedList is provided outside in this version.
  IterativeSearchState<dist_t> Open(const void *query, int k, VisitedList *vl = nullptr, int batch_k = 1) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k, batch_k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      state.result_set_.emplace(-curr_dist, curr_obj);
      state.top_candidates_.emplace(curr_dist, curr_obj);

      UpdateNext(&state);
      state.has_ran_ = true;
    }
    return state;
  }

  // VistedList is provided outside in this version.
  IterativeSearchState<dist_t>
  OpenFiltered(const void *query, int k, VisitedList *vl = nullptr, BaseFilterFunctor *pred = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      if (pred == nullptr || (*pred)(curr_obj)) {
        state.result_set_.emplace(-curr_dist, curr_obj);
      }
      state.top_candidates_.emplace(curr_dist, curr_obj);

      // UpdateNextFiltered(&state, pred);
    }
    return state;
  }

  IterativeSearchState<dist_t> OpenUninit(const void *query, int k, VisitedList *vl = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
    return state;
  }

  // VistedList is provided outside in this version.
  IterativeSearchState<dist_t>
  OpenTwoHop(const void *query, int k, RangeQuery<float> *pred, VisitedList *vl = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      if (pred == nullptr || (*pred)(curr_obj)) {
        state.result_set_.emplace(-curr_dist, curr_obj);
      }
      state.top_candidates_.emplace(curr_dist, curr_obj);

      UpdateNextTwoHop(&state, pred);
      state.has_ran_ = true;
    }
    return state;
  }

  IterativeSearchState<dist_t>
  OpenTwoHop(const void *query, int k, BaseFilterFunctor *bitset, VisitedList *vl = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      if (bitset == nullptr || (*bitset)(curr_obj)) {
        state.result_set_.emplace(-curr_dist, curr_obj);
      }
      state.top_candidates_.emplace(curr_dist, curr_obj);

      UpdateNextTwoHop(&state, bitset);
      state.has_ran_ = true;
    }
    return state;
  }

  // VistedList is provided outside in this version.
  IterativeSearchState<dist_t> OpenNavix(const void *query, int k, RangeQuery<float> *pred, VisitedList *vl = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      if (pred == nullptr || (*pred)(curr_obj)) {
        state.result_set_.emplace(-curr_dist, curr_obj);
      }
      state.top_candidates_.emplace(curr_dist, curr_obj);

      UpdateNextNavix(&state, pred);
    }
    return state;
  }

  IterativeSearchState<dist_t> OpenNeo(const void *query, int k, VisitedList *vl = nullptr) {
    hnsw_->setEf(this->initial_efs_);
    IterativeSearchState<dist_t> state(query, k);
    state.vl_ = vl ? vl : hnsw_->visited_list_pool_->getFreeVisitedList();
    state.ncomp_ = 0;
    state.total_ = 0;
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
            state.ncomp_++;

            if (d < curr_dist) {
              curr_dist = d;
              curr_obj = cand;
              changed = true;
            }
          }
        }
      }
      state.vl_->mass[curr_obj] = state.vl_->curV;
      state.candidate_set_.emplace(-curr_dist, curr_obj);
      state.result_set_.emplace(-curr_dist, curr_obj);
      // state.otree_.insert(std::make_pair(curr_dist, curr_obj));
      state.btree_[curr_dist] = std::make_pair(curr_dist, curr_obj);

      UpdateNextNeo(&state);
    }
    return std::move(state);
  }

  pair<dist_t, labeltype> Next(IterativeSearchState<dist_t> *state) {
    if (state->total_ >= state->k_) {
      return {-1, -1};
    }
    if (state->has_ran_ && state->result_set_.empty()) {
      state->has_ran_ = false;
      return {-1, -1};
    }
    if (state->has_ran_ && !state->result_set_.empty() && state->cur_cnt < state->batch_k_) {
      auto top = state->result_set_.top();
      state->result_set_.pop();
      state->total_++;
      state->cur_cnt++;
      if (state->cur_cnt == state->batch_k_) {
        state->has_ran_ = false;
        state->cur_cnt = 0;
      }
      return {-top.first, top.second};
    } else {
      UpdateNext(state);
      state->has_ran_ = true;
      return Next(state);
    }
  }

  priority_queue<pair<dist_t, labeltype>> NextBatch(IterativeSearchState<dist_t> *state) {
    if (state->total_ >= state->k_) {
      return {};
    }
    if (state->has_ran_ && !state->result_set_.empty()) {
      state->has_ran_ = false;
      return {};
    } else {
      UpdateNext(state);
      state->has_ran_ = true;
      return {};
    }
  }

  priority_queue<pair<dist_t, labeltype>>
  NextBatchTwoHop(IterativeSearchState<dist_t> *state, RangeQuery<float> *pred) {
    if (state->total_ >= state->k_) {
      return {};
    }
    if (state->has_ran_ && !state->result_set_.empty()) {
      state->has_ran_ = false;
    } else {
      int cnt = UpdateNextTwoHop(state, pred);
      state->has_ran_ = true;
    }
    return {};
    // return NextBatchTwoHop(state, pred);
  }

  priority_queue<pair<dist_t, labeltype>>
  NextBatchTwoHop(IterativeSearchState<dist_t> *state, BaseFilterFunctor *bitset) {
    if (state->total_ >= state->k_) {
      return {};
    }
    if (state->has_ran_ && !state->result_set_.empty()) {
      state->has_ran_ = false;
    } else {
      int cnt = UpdateNextTwoHop(state, bitset);
      state->has_ran_ = true;
    }
    return {};
    // return NextBatchTwoHop(state, pred);
  }

  priority_queue<pair<dist_t, labeltype>> NextBatchNavix(IterativeSearchState<dist_t> *state, RangeQuery<float> *pred) {
    if (state->total_ >= state->k_) {
      return {};
    }
    if (HasNext(state)) {
      auto moved = std::move(state->batch_rz_);
      state->total_ += moved.size();
      state->batch_rz_ = {};
      return moved;
    } else {
      int cnt = UpdateNextNavix(state, pred);
      if (cnt == 0) {
        return {};
      }
    }
    return NextBatchNavix(state, pred);
  }

  priority_queue<pair<dist_t, labeltype>>
  NextBatchFiltered(IterativeSearchState<dist_t> *state, BaseFilterFunctor *pred) {
    if (state->total_ >= state->k_) {
      return {};
    }
    if (HasNext(state)) {
      auto moved = std::move(state->batch_rz_);
      state->total_ += moved.size();
      state->batch_rz_ = {};
      return moved;
    } else {
      int cnt = UpdateNextFiltered(state, pred);
      if (cnt == 0) {
        return {};
      }
    }
    return NextBatchFiltered(state, pred);
  }

  pair<dist_t, labeltype> NextNeo(IterativeSearchState<dist_t> *state) {
    if (state->total_ >= state->k_) {
      return {-1, -1};
    }
    if (HasNext(state)) {
      auto top = state->batch_rz_.top();
      state->batch_rz_.pop();
      state->total_++;
      return {-top.first, top.second};
    } else {
      int cnt = UpdateNextNeo(state);
      if (cnt == 0) {
        return {-1, -1};
      }
    }
    return NextNeo(state);
  }

  priority_queue<pair<dist_t, labeltype>> NextBatchNeo(IterativeSearchState<dist_t> *state) {
    if (state->total_ >= state->k_) {
      return {};
    }
    // if (HasNext(state)) {
    //   auto moved = std::move(state->batch_rz_);
    //   state->total_ += moved.size();
    //   state->batch_rz_ = {};
    //   return moved;
    // } else {
    //   int cnt = UpdateNextNeo(state);
    //   if (cnt == 0) {
    //     return {};
    //   }
    // }
    if (!HasNext(state)) {
      int cnt = UpdateNextNeo(state);
      if (cnt == 0) {
        return {};
      }
    }
    auto moved = std::move(state->batch_rz_);
    state->total_ += moved.size();
    state->batch_rz_ = {};
    return moved;
    // return NextBatchNeo(state);
  }

  void Close(IterativeSearchState<dist_t> *state) { hnsw_->setEf(this->initial_efs_); }

  int GetNcomp(IterativeSearchState<dist_t> *state) { return state->ncomp_; }

  void SetSearchParam(int batch_k, int delta_efs) {
    this->batch_k_ = batch_k;
    this->delta_efs_ = delta_efs;
    this->initial_efs_ = std::max(batch_k, delta_efs);
    hnsw_->setEf(this->initial_efs_);
  }

  void SetSearchParam(int batch_k, int initial_efs, int delta_efs) {
    this->batch_k_ = batch_k;
    this->delta_efs_ = delta_efs;
    this->initial_efs_ = initial_efs;
    hnsw_->setEf(this->initial_efs_);
  }
};