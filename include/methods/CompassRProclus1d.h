#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "../../thirdparty/btree/btree_map.h"
#include "../hnswlib/hnswlib.h"
#include "../utils/predicate.h"
#include "utils/Pod.h"
#include "Proclus.h"
#include "methods/basis/ReentrantHNSW.h"
#include "faiss/MetricType.h"

namespace fs = boost::filesystem;
using coroutine_t = boost::coroutines2::coroutine<int>;

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassRProclus1d {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  Proclus *proclus_;
  // faiss::IndexIVFPQ ivfpq_;

  HierarchicalNSW<dist_t> cgraph_;

  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

  int dim_;
  int nlist_;

  faiss::idx_t *ranked_clusters_;

 public:
  CompassRProclus1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist, Proclus *proclus)
      : space_(d),
        hnsw_(&space_, max_elements, M, efc),
        proclus_(proclus),
        attrs_(max_elements, std::numeric_limits<attr_t>::max()),
        btrees_(nlist, btree::btree_map<attr_t, labeltype>()),
        cgraph_(&space_, nlist, 8, 100),
        dim_(d),
        nlist_(nlist) {}
  int AddPoint(const void *data_point, labeltype label, attr_t attr);
  int AddGraphPoint(const void *data_point, labeltype label);
  int AddPointsToIvf(size_t n, const void *data, labeltype *labels, attr_t *attrs);
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
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = nlist_ / 20;
    proclus_->search_l1_rerank_l2(nq, (float *)query, ranked_clusters, distances, nprobe);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;
      auto itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
      auto itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);

      while (true) {
        int crel = 0;
        // if (candidate_set.empty() || distances[curr_ci] < -candidate_set.top().first) {
        if (candidate_set.empty() || (curr_ci < (q + 1) * nprobe && -candidate_set.top().first > distances[curr_ci])) {
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
            // visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * dim_, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          metrics[q].nround++;
          int cnt = hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second]) continue;
            visited[top.second] = true;
            metrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        hnsw_.ReentrantSearchKnnBounded(
            (float *)query + q * dim_,
            k,
            distances[curr_ci],
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
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      // vector<std::pair<dist_t, labeltype>> result(sz);
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }

  void LoadClusterGraph(fs::path path) { this->cgraph_->loadIndex(path.string(), &this->space_); }

  void LoadRanking(fs::path path, attr_t *attrs) {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    // int32_t assigned_cluster;
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(assigned_cluster));
      attrs_[i] = attrs[i];
      btrees_[assigned_cluster].insert(std::make_pair(attrs[i], (labeltype)i));
    }
  }

  void BuildClusterGraph() {
    auto centroids = proclus_->medoids;
    for (int i = 0; i < nlist_; i++) {
      this->cgraph_->addPoint(centroids + i * dim_, i);
    }
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cgraph_->saveIndex(path.string());
  }

  void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      out.write((char *)(ranked_clusters_ + i), sizeof(faiss::idx_t));
    }
  }
};

template <typename dist_t, typename attr_t>
int CompassRProclus1d<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassRProclus1d<dist_t, attr_t>::AddPointsToIvf(size_t n, const void *data, labeltype *labels, attr_t *attr) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  ranked_clusters_ = new faiss::idx_t[n];
  float *distances = new float[n];
  proclus_->search_l1_rerank_l2(n, (float *)data, ranked_clusters_, distances);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attr[i];
    btrees_[ranked_clusters_[i]].insert(std::make_pair(attr[i], labels[i]));
  }

  delete[] distances;
  return n;
}
