#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>
#include "Pod.h"
#include "btree_map.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "methods/ReentrantHNSW.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
using coroutine_t = boost::coroutines2::coroutine<int>;

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassR1d {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  HierarchicalNSW<dist_t> cgraph_;

  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

  faiss::idx_t *ranked_clusters_;

 public:
  CompassR1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist);
  int AddPoint(const void *data_point, labeltype label, attr_t attr);
  int AddGraphPoint(const void *data_point, labeltype label);
  int AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attrs);
  void TrainIvf(size_t n, const void *data);
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = 100;  // TOREVERT
    auto ranked_clusters = new faiss::idx_t[nq * nprobe];
    auto distances = new float[nq * nprobe];
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> cluster_set;

      for (int i = 0; i < nprobe; i++) {
        if (ranked_clusters[q * nprobe + i] == -1) break;
        cluster_set.emplace(-distances[q * nprobe + i], ranked_clusters[q * nprobe + i]);
      }
      vector<bool> visited(hnsw_.cur_element_count, false);

      vector<coroutine_t::pull_type> functions;
      functions.reserve(nprobe);
      std::unordered_map<faiss::idx_t, int> map;
      auto itr = functions.begin();
      for (int i = 0; i < nprobe; i++, itr++) {
        auto cluster_push = [&, i](coroutine_t::push_type &push) {
          push(0);
          int crel = 0;
          size_t efs = std::max((size_t)std::abs(k), hnsw_.ef_);

          auto cluster = ranked_clusters[q * nprobe + i];
          if (cluster == -1) push(crel);
          auto rel_beg = btrees_[cluster].lower_bound(l_bound);
          auto rel_end = btrees_[cluster].upper_bound(u_bound);
          auto targets = vector<std::pair<attr_t, labeltype>>(rel_beg, rel_end);
          // std::random_shuffle(targets.begin(), targets.end());

          for (size_t i = 0; i < targets.size(); i++) {
            auto label = targets[i].second;
            // assert(hnsw_.label_lookup_.find(label) != hnsw_.label_lookup_.end());
            auto tableid = hnsw_.label_lookup_[label];
            assert(label == tableid);
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((dist_t *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            if (top_candidates.size() < efs || dist < upper_bound) {
              candidate_set.emplace(-dist, tableid);
              top_candidates.emplace(dist, tableid);
              metrics[q].is_ivf_ppsl[label] = true;
              if (top_candidates.size() > efs_) top_candidates.pop();
              if (++crel >= nrel) {
                push(crel);
                crel = 0;
              }
            }
          }
          push(crel);
        };
        coroutine_t::pull_type coroutine(cluster_push);
        functions.push_back(std::move(coroutine));
        // functions.emplace_back(coroutine_t::pull_type(cluster_push));
        // functions.emplace(std::piecewise_construct, std::forward_as_tuple(ranked_clusters[i]),
        // std::forward_as_tuple(cluster_push));
        map.emplace(ranked_clusters[i], i);
      }

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      size_t total_proposed = 0;
      size_t max_dist_comp = 10;
      metrics[q].nround = 0;

      while (true) {
        // break;  // TOREVERT
        if (candidate_set.empty() || (candidate_set.top().first < cluster_set.top().first)) {
          auto cluster = cluster_set.top().second;
          if (functions[map[cluster]]) {
            using namespace std::chrono;
            auto context_start = steady_clock::now();
            (functions[map[cluster]])();
            total_proposed += (functions[map[cluster]]).get();
            auto context_stop = steady_clock::now();
            metrics[q].nround++;
          } else {
            cluster_set.pop();
          }
        }
        hnsw_.ReentrantSearchKnn(
            (float *)query + q * ivf_->d,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        if (top_candidates.size() >= efs_) break;
      }

      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      // vector<std::pair<dist_t, labeltype>> result(sz);
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    delete[] ranked_clusters;
    delete[] distances;
    return results;
  }

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV1(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    RangeQuery<float> pred(l_bound, u_bound, &attrs_);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;
      auto itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
      auto itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
      float stop_bound = 1e10;

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1) && -candidate_set.top().first > distances[curr_ci])) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
                itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
                stop_bound = 1e10;
                // recycle_set = priority_queue<pair<float, int64_t>>();
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            recycle_set.emplace(-dist, tableid);

          }
          metrics[q].nround++;
          int cnt = hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() >= efs_) top_candidates.pop(); // better not to overflow the result queue
            recycle_set.pop();
            cnt--;
          }
        }

        hnsw_.ReentrantSearchKnnBounded(
            (float *)query + q * ivf_->d,
            k,
            // -recycle_set.top().first, // cause infinite loop?
            1e10,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV2(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    RangeQuery<float> pred(l_bound, u_bound, &attrs_);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;
      auto itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
      auto itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);

      int cnt = 0;
      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1) && -candidate_set.top().first > distances[curr_ci])) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
                itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            candidate_set.emplace(-dist, tableid);
            if (dist < upper_bound) {
              top_candidates.emplace(dist, tableid);
              if (top_candidates.size() > efs_) top_candidates.pop();
            } else {
              recycle_set.emplace(-dist, tableid);
            }
          }
          metrics[q].nround++;
        }

        hnsw_.ReentrantSearchKnn(
            (float *)query + q * ivf_->d,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0) ||
        //     curr_ci >= (q + 1) * nprobe) {
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  //   vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV3(
  //       const void *query,
  //       const int nq,
  //       const int k,
  //       const attr_t &l_bound,
  //       const attr_t &u_bound,
  //       const int efs,
  //       const int nrel,
  //       const int nthread,
  //       vector<Metric> &metrics
  //   ) {
  //     auto efs_ = std::max(k, efs);
  //     hnsw_.setEf(efs_);
  //     int nprobe = ivf_->nlist;
  //     faiss::idx_t *ranked_clusters;
  //     // this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

  //     vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

  //     // #pragma omp parallel for num_threads(nthread) schedule(static)
  //     for (int q = 0; q < nq; q++) {
  //       priority_queue<pair<float, int64_t>> top_candidates;
  //       priority_queue<pair<float, int64_t>> candidate_set;

  //       vector<bool> visited(hnsw_.cur_element_count, false);

  //       RangeQuery<float> pred(l_bound, u_bound, &attrs_);
  //       metrics[q].nround = 0;
  //       metrics[q].ncomp = 0;

  //       {
  //         tableint currObj = hnsw_.enterpoint_node_;
  //         dist_t currDist = hnsw_.fstdistfunc_(
  //             (float *)query + q * ivf_->d, hnsw_.getDataByInternalId(hnsw_.enterpoint_node_), hnsw_.dist_func_param_
  //         );

  //         for (int level = hnsw_.maxlevel_; level > 0; level--) {
  //           bool changed = true;
  //           while (changed) {
  //             changed = false;
  //             unsigned int *data;

  //             data = (unsigned int *)hnsw_.get_linklist(currObj, level);
  //             int size = hnsw_.getListCount(data);
  //             metrics[q].ncomp += size;

  //             tableint *datal = (tableint *)(data + 1);
  //             for (int i = 0; i < size; i++) {
  //               tableint cand = datal[i];

  //               if (cand < 0 || cand > hnsw_.max_elements_) throw std::runtime_error("cand error");
  //               dist_t d = hnsw_.fstdistfunc_(
  //                   (float *)query + q * ivf_->d, hnsw_.getDataByInternalId(cand), hnsw_.dist_func_param_
  //               );

  //               if (d < currDist) {
  //                 currDist = d;
  //                 currObj = cand;
  //                 changed = true;
  //               }
  //             }
  //           }
  //         }
  //         ranked_clusters = ranked_clusters_ + currObj * nprobe;
  //         visited[currObj] = true;
  //         candidate_set.emplace(-currDist, currObj);
  //         top_candidates.emplace(currDist, currObj);
  //       }

  //       int curr_ci = 0;
  //       auto itr_beg = btrees_[ranked_clusters[0]].lower_bound(l_bound);
  //       auto itr_end = btrees_[ranked_clusters[0]].upper_bound(u_bound);
  //       //       while (itr_beg != itr_end) {
  //       //         tableint id = (*itr_beg).second;
  //       //         itr_beg++;
  //       //         visited[id] = true;
  //       // #ifdef USE_SSE
  //       //         _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
  //       // #endif
  //       //         auto vect = hnsw_.getDataByInternalId(id);
  //       //         auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
  //       //         candidate_set.emplace(-dist, id);
  //       //         top_candidates.emplace(dist, id);
  //       //       }
  //       //       curr_ci = 1;

  //       int cnt = 0;
  //       while (true) {
  //         int crel = 0;
  //         // if (candidate_set.empty() || distances[curr_ci] < -candidate_set.top().first) {
  //         if (candidate_set.empty() ||
  //             (!top_candidates.empty() && -candidate_set.top().first > top_candidates.top().first)) {
  //           while (crel < nrel) {
  //             if (itr_beg == itr_end) {
  //               if (++curr_ci == nprobe)
  //                 break;
  //               else {
  //                 itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
  //                 itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
  //                 continue;
  //               }
  //             }

  //             auto tableid = (*itr_beg).second;
  //             itr_beg++;
  // #ifdef USE_SSE
  //             _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
  // #endif
  //             if (visited[tableid]) continue;
  //             visited[tableid] = true;

  //             auto vect = hnsw_.getDataByInternalId(tableid);
  //             auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
  //             metrics[q].ncomp++;
  //             crel++;

  //             auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() :
  //             top_candidates.top().first; if (top_candidates.size() < efs || dist < upper_bound) {
  //               candidate_set.emplace(-dist, tableid);
  //               top_candidates.emplace(dist, tableid);
  //               metrics[q].is_ivf_ppsl[tableid] = true;
  //               if (top_candidates.size() > efs_) top_candidates.pop();
  //             }
  //           }
  //           metrics[q].nround++;
  //         }

  //         hnsw_.ReentrantSearchKnn(
  //             (float *)query + q * ivf_->d,
  //             k,
  //             -1,
  //             top_candidates,
  //             candidate_set,
  //             visited,
  //             &pred,
  //             std::ref(metrics[q].ncomp),
  //             std::ref(metrics[q].is_graph_ppsl)
  //         );
  //         // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0) ||
  //         //     curr_ci >= (q + 1) * nprobe) {
  //         if ((top_candidates.size() >= efs_) || curr_ci >= nprobe) {
  //           break;
  //         }
  //       }

  //       while (top_candidates.size() > k) top_candidates.pop();
  //       size_t sz = top_candidates.size();
  //       // vector<std::pair<dist_t, labeltype>> result(sz);
  //       while (!top_candidates.empty()) {
  //         results[q][--sz] = top_candidates.top();
  //         top_candidates.pop();
  //       }
  //     }

  //     return results;
  //   }

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV4(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    // this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;
      vector<pair<float, labeltype>> clusters = cgraph_.searchKnnCloserFirst((float *)(query) + q * ivf_->d, nprobe);

      vector<bool> visited(hnsw_.cur_element_count, false);

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = 0;
      auto itr_beg = btrees_[clusters[curr_ci].second].lower_bound(l_bound);
      auto itr_end = btrees_[clusters[curr_ci].second].upper_bound(u_bound);

      int cnt = 0;
      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe && -candidate_set.top().first > clusters[curr_ci].first)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= nprobe)
                break;
              else {
                itr_beg = btrees_[clusters[curr_ci].second].lower_bound(l_bound);
                itr_end = btrees_[clusters[curr_ci].second].upper_bound(u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            candidate_set.emplace(-dist, tableid);
            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            if (dist < upper_bound) {
              top_candidates.emplace(dist, tableid);
              if (top_candidates.size() > efs_) top_candidates.pop();
            } else {
              recycle_set.emplace(-dist, tableid);
            }
          }
          metrics[q].nround++;
        }

        hnsw_.ReentrantSearchKnn(
            (float *)query + q * ivf_->d,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0) ||
        //     curr_ci >= (q + 1) * nprobe) {
        if ((top_candidates.size() >= efs_) || curr_ci >= nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV5(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    // this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;
      vector<pair<float, labeltype>> clusters = cgraph_.searchKnnCloserFirst((float *)(query) + q * ivf_->d, nprobe);

      vector<bool> visited(hnsw_.cur_element_count, false);

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = 0;
      auto itr_beg = btrees_[clusters[curr_ci].second].lower_bound(l_bound);
      auto itr_end = btrees_[clusters[curr_ci].second].upper_bound(u_bound);

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe && -candidate_set.top().first > clusters[curr_ci].first)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= nprobe)
                break;
              else {
                itr_beg = btrees_[clusters[curr_ci].second].lower_bound(l_bound);
                itr_end = btrees_[clusters[curr_ci].second].upper_bound(u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          metrics[q].nround++;
          int cnt = hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() >= efs_) top_candidates.pop(); // better not to overflow the result queue
            recycle_set.pop();
            cnt--;
          }
        }

        hnsw_.ReentrantSearchKnnBounded(
            (float *)query + q * ivf_->d,
            k,
            // clusters[curr_ci].first,
            1e10,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0) ||
        //     curr_ci >= (q + 1) * nprobe) {
        if ((top_candidates.size() >= efs_) || curr_ci >= nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }

  void LoadClusterGraph(fs::path path) { this->cgraph_.loadIndex(path.string(), &this->space_); }

  void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
  }

  void LoadRanking(fs::path path, attr_t *attrs) {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      attrs_[i] = attrs[i];
      btrees_[assigned_cluster].insert(std::make_pair(attrs[i], (labeltype)i));
    }
  }

  void BuildClusterGraph() {
    auto centroids = ((faiss::IndexFlatL2 *)this->ivf_->quantizer)->get_xb();
    for (int i = 0; i < ivf_->nlist; i++) {
      this->cgraph_.addPoint(centroids + i * ivf_->d, i);
    }
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }

  void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
  }

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cgraph_.saveIndex(path.string());
  }

  void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      out.write((char *)(ranked_clusters_ + i), sizeof(faiss::idx_t));
    }
  }
};

template <typename dist_t, typename attr_t>
CompassR1d<dist_t, attr_t>::CompassR1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist)
    : space_(d),
      hnsw_(&space_, max_elements, M, efc),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, std::numeric_limits<attr_t>::max()),
      btrees_(nlist, btree::btree_map<attr_t, labeltype>()),
      cgraph_(&space_, nlist, 8, 100) {
  ivf_->nprobe = nlist;
}

template <typename dist_t, typename attr_t>
int CompassR1d<dist_t, attr_t>::AddPoint(const void *data_point, labeltype label, attr_t attr) {
  hnsw_.addPoint(data_point, label, -1);
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes

  faiss::idx_t assigned_cluster;
  quantizer_.assign(1, (float *)data_point, &assigned_cluster, 1);
  btrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassR1d<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassR1d<dist_t, attr_t>::AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attr) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  ranked_clusters_ = new faiss::idx_t[n];
  ivf_->quantizer->assign(n, (float *)data, ranked_clusters_);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attr[i];
    btrees_[ranked_clusters_[i]].insert(std::make_pair(attr[i], labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassR1d<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivf_->add(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // auto assigned_clusters = new faiss::idx_t[n * 1];
  // ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
}
