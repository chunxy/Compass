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
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "Pod.h"
#include "btree_map.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/ReentrantHNSW.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
using coroutine_t = boost::coroutines2::coroutine<int>;

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassF1d {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;
  faiss::idx_t *assigned_clusters;

  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

 public:
  CompassF1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist);
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
      const int min_comp,
      const int nthread,
      vector<Metric> &metrics
  );

  void SaveGraph(fs::path path);
  void SaveIvf(fs::path path);
  void LoadGraph(fs::path path);
  void LoadIvf(fs::path path);
};

template <typename dist_t, typename attr_t>
CompassF1d<dist_t, attr_t>::CompassF1d(
    size_t d,
    size_t M,
    size_t efc,
    size_t max_elements,
    size_t nlist
)
    : space_(d),
      hnsw_(&space_, max_elements, M, efc),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, std::numeric_limits<attr_t>::max()),
      btrees_(nlist, btree::btree_map<attr_t, labeltype>()) {
  ivf_->nprobe = nlist;
}

template <typename dist_t, typename attr_t>
int CompassF1d<dist_t, attr_t>::AddPoint(const void *data_point, labeltype label, attr_t attr) {
  hnsw_.addPoint(data_point, label, -1);
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes

  faiss::idx_t assigned_cluster;
  quantizer_.assign(1, (float *)data_point, &assigned_cluster, 1);
  btrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassF1d<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassF1d<dist_t, attr_t>::AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attr) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  assigned_clusters = new faiss::idx_t[n * 1];
  ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attr[i];
    btrees_[assigned_clusters[i]].insert(std::make_pair(attr[i], labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassF1d<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivf_->add(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // auto assigned_clusters = new faiss::idx_t[n * 1];
  // ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
}

template <typename dist_t, typename attr_t>
vector<vector<pair<float, hnswlib::labeltype>>> CompassF1d<dist_t, attr_t>::SearchKnn(
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

  vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));
  vector<decltype(btrees_[0].lower_bound(0))> itr_begs(ivf_->nlist);
  vector<decltype(btrees_[0].upper_bound(0))> itr_ends(ivf_->nlist);

  // #pragma omp parallel for num_threads(nthread) schedule(static)
  for (int q = 0; q < nq; q++) {
    priority_queue<pair<float, int64_t>> top_candidates;
    priority_queue<pair<float, int64_t>> candidate_set;
    priority_queue<pair<float, int64_t>> frontiers;

    vector<bool> visited(hnsw_.cur_element_count, false);
    for (int i = 0; i < ivf_->nlist; i++) {
      itr_begs[i] = btrees_[i].lower_bound(l_bound);
      itr_ends[i] = btrees_[i].upper_bound(u_bound);
    }

    RangeQuery<float> pred(l_bound, u_bound, &attrs_);
    size_t total_proposed = 0;
    size_t max_dist_comp = 10;
    metrics[q].nround = 0;

    hnsw_.ReentrantSearchKnnV2(
        (float *)query + q * ivf_->d,
        k,
        -1,
        top_candidates,
        candidate_set,
        frontiers,
        visited,
        &pred,
        metrics[q].ncomp,
        metrics[q].is_graph_ppsl,
        true
    );

    while (true) {
      int crel = 0;
      while (!frontiers.empty()) {
        auto top = frontiers.top();
        frontiers.pop();
        faiss::idx_t c = assigned_clusters[top.second];
        while (itr_begs[c] != itr_ends[c]) {
          if (!visited[(*itr_begs[c]).second]) {
            auto tableid = (*itr_begs[c]).second;
            visited[tableid] = true;
            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            // auto upper_bound =
            //     top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            // if (top_candidates.size() < efs || dist < upper_bound) {
            //   if (top_candidates.size() > efs_) top_candidates.pop();
            // }
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            candidate_set.emplace(-dist, tableid);
            top_candidates.emplace(dist, tableid);
            crel++;
          }
          itr_begs[c]++;
        }
        if (crel > nrel) {
          metrics[q].nround++;
          break;
        }
      }
      if (candidate_set.empty()) {
        break;
      }
      hnsw_.ReentrantSearchKnnV2(
          (float *)query + q * ivf_->d,
          k,
          -1,
          top_candidates,
          candidate_set,
          frontiers,
          visited,
          &pred,
          std::ref(metrics[q].ncomp),
          std::ref(metrics[q].is_graph_ppsl)
      );
      // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0)) break;
      if ((top_candidates.size() >= efs_)) break;
    }

    while (top_candidates.size() > k) top_candidates.pop();
    size_t sz = top_candidates.size();
    while (!top_candidates.empty()) {
      results[q][--sz] = top_candidates.top();
      top_candidates.pop();
    }
  }

  return results;
}

template <typename dist_t, typename attr_t>
void CompassF1d<dist_t, attr_t>::LoadGraph(fs::path path) {
  this->hnsw_.loadIndex(path.string(), &this->space_);
}

template <typename dist_t, typename attr_t>
void CompassF1d<dist_t, attr_t>::LoadIvf(fs::path path) {
  auto ivf_file = fopen(path.c_str(), "r");
  auto index = faiss::read_index(ivf_file);
  this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
}

template <typename dist_t, typename attr_t>
void CompassF1d<dist_t, attr_t>::SaveGraph(fs::path path) {
  fs::create_directories(path.parent_path());
  this->hnsw_.saveIndex(path.string());
}

template <typename dist_t, typename attr_t>
void CompassF1d<dist_t, attr_t>::SaveIvf(fs::path path) {
  fs::create_directories(path.parent_path());
  faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
}
