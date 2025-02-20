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
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPQ.h"
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
class CompassRImi1d {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  faiss::MultiIndexQuantizer quantizer_;
  // faiss::IndexPQ *imi_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

  // config
  size_t nrel_;

 public:
  CompassRImi1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nsub, size_t nbits, size_t nrel);
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
CompassRImi1d<dist_t, attr_t>::CompassRImi1d(
    size_t d,
    size_t M,
    size_t efc,
    size_t max_elements,
    size_t nsub,
    size_t nbits,
    size_t nrel
)
    : space_(d),
      hnsw_(&space_, max_elements, M, efc),
      quantizer_(d, nsub, nbits),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, 1 << (nsub * nbits))),
      attrs_(max_elements, std::numeric_limits<attr_t>::max()),
      btrees_((1 << (nbits * nsub)), btree::btree_map<attr_t, labeltype>()),
      nrel_(nrel) {
  // pq_->nprobe = nlist;
  ivf_->quantizer_trains_alone = true;
}

template <typename dist_t, typename attr_t>
int CompassRImi1d<dist_t, attr_t>::AddPoint(const void *data_point, labeltype label, attr_t attr) {
  hnsw_.addPoint(data_point, label, -1);
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes

  faiss::idx_t assigned_cluster;
  ivf_->assign(1, (float *)data_point, &assigned_cluster, 1);
  btrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassRImi1d<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassRImi1d<dist_t, attr_t>::AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attr) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  auto assigned_clusters = new faiss::idx_t[n * 1];
  ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attr[i];
    btrees_[assigned_clusters[i]].insert(std::make_pair(attr[i], labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassRImi1d<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  ivf_->add(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // auto assigned_clusters = new faiss::idx_t[n * 1];
  // ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
}

template <typename dist_t, typename attr_t>
vector<vector<pair<float, hnswlib::labeltype>>> CompassRImi1d<dist_t, attr_t>::SearchKnn(
    const void *query,
    const int nq,
    const int k,
    const attr_t &l_bound,
    const attr_t &u_bound,
    const int efs,
    const int min_comp,
    const int nthread,
    vector<Metric> &metrics
) {
  auto efs_ = std::max(k, efs);
  hnsw_.setEf(efs_);
  // int nprobe = 1 << (ivf_->pq.M * ivf_->pq.nbits);
  int nprobe = 1000;
  // auto centroids = quantizer_.get_xb();
  // auto dist_func = quantizer_.get_distance_computer();
  auto ranked_clusters = new faiss::idx_t[nq * nprobe];
  auto distances = new float[nq * nprobe];
  this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);


  vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

  // #pragma omp parallel for num_threads(nthread) schedule(static)
  for (int q = 0; q < nq; q++) {
    priority_queue<pair<float, int64_t>> top_candidates;
    priority_queue<pair<float, int64_t>> candidate_set;

    vector<bool> visited(hnsw_.cur_element_count, false);

    // auto first_cluster = cluster_set.top().second;
    auto curr_ci = q * nprobe;
    auto itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
    auto itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);

    RangeQuery<float> pred(l_bound, u_bound, &attrs_);
    size_t total_proposed = 0;
    size_t max_dist_comp = 10;
    metrics[q].nround = 0;
    metrics[q].tcontext = 0;

    while (true) {
      int crel = 0;
      if (candidate_set.empty() || distances[curr_ci] < -candidate_set.top().first) {
        while (crel < nrel_) {
          if (itr_beg == itr_end) {
            if (++curr_ci == (q + 1) * nprobe)
              break;
            else {
              itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
              itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
              continue;
            }
          }
          // auto label = (*itr_beg).second;
          // auto tableid = hnsw_.label_lookup_[label];
          // assert(label == tableid);
          auto tableid = (*itr_beg).second;
          itr_beg++;
          if (visited[tableid]) continue;
          visited[tableid] = true;

          auto vect = hnsw_.getDataByInternalId(tableid);
          auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
          metrics[q].ncomp++;
          crel++;

          auto upper_bound =
              top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
          if (top_candidates.size() < efs || dist < upper_bound) {
            candidate_set.emplace(-dist, tableid);
            top_candidates.emplace(dist, tableid);
            metrics[q].is_ivf_ppsl[tableid] = true;
            // metrics[q].is_ivf_ppsl[label] = true;
            if (top_candidates.size() > efs_) top_candidates.pop();
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
      if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp < 0) || curr_ci >= (q + 1) * nprobe)
        break;
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

template <typename dist_t, typename attr_t>
void CompassRImi1d<dist_t, attr_t>::LoadGraph(fs::path path) {
  this->hnsw_.loadIndex(path.string(), &this->space_);
}

template <typename dist_t, typename attr_t>
void CompassRImi1d<dist_t, attr_t>::LoadIvf(fs::path path) {
  auto ivf_file = fopen(path.c_str(), "r");
  auto index = faiss::read_index(ivf_file);
  this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
}

template <typename dist_t, typename attr_t>
void CompassRImi1d<dist_t, attr_t>::SaveGraph(fs::path path) {
  fs::create_directories(path.parent_path());
  this->hnsw_.saveIndex(path.string());
}

template <typename dist_t, typename attr_t>
void CompassRImi1d<dist_t, attr_t>::SaveIvf(fs::path path) {
  fs::create_directories(path.parent_path());
  faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
}
