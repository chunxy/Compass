#pragma once

#include <fmt/core.h>
#include <boost/filesystem.hpp>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "btree_map.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/Pod.h"

using hnswlib::L2Space;
using hnswlib::labeltype;
using std::pair;
using std::vector;

namespace fs = boost::filesystem;

template <typename dist_t, typename attr_t>
class CompassIvf1D {
 private:
  L2Space space_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  // vector<vector<attr_t>> attrs_;
  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

  const float *xb_;

 public:
  CompassIvf1D(size_t d, size_t max_elements, size_t nlist, size_t nprobe, const float *xb);
  int Add(const void *data_point, labeltype label, attr_t attr);
  // void AddMultiple(size_t n, const void *data, labeltype *labels, attr_t
  // *attrs);
  int AddIvfPoints(size_t n, const void *data, labeltype *labels, const vector<attr_t> &attrs);
  void TrainIvf(size_t n, const void *data);

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int nprobe,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters
  ) {
    // auto centroids = quantizer_.get_xb();
    // auto dist_func = quantizer_.get_distance_computer();
    ivf_->quantizer->assign(nq, (float *)query, ranked_clusters, nprobe);
    // auto &dm = ivf_->direct_map;

    vector<vector<pair<float, labeltype>>> result(nq, vector<pair<float, labeltype>>(k));
    for (int q = 0; q < nq; q++) {
      metrics[q].ncomp = 0;

      std::priority_queue<pair<float, labeltype>> top_candidates;
      int i = 0;
      for (i = 0; i < nprobe; i++) {
        auto cluster = ranked_clusters[q * nprobe + i];
        if (cluster == -1) break;
        auto rel_beg = btrees_[cluster].lower_bound(l_bound);
        auto rel_end = btrees_[cluster].upper_bound(u_bound);
        while (rel_beg != rel_end) {
          auto j = (*rel_beg).second;
          metrics[q].is_ivf_ppsl[j] = true;
          const dist_t *vect = xb_ + j * quantizer_.d;
          auto dist =
              space_.get_dist_func()((dist_t *)query + q * ivf_->d, vect, space_.get_dist_func_param());
          metrics[q].ncomp++;
          top_candidates.emplace(dist, j);
          rel_beg++;
        }
      }
      metrics[q].nround = i;

      while (top_candidates.size() > k) top_candidates.pop();
      int sz = top_candidates.size();
      result[q].resize((sz));
      while (!top_candidates.empty()) {
        result[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return result;
  }

  void SaveIvf(fs::path path);
  void LoadIvf(fs::path path);
};

template <typename dist_t, typename attr_t>
CompassIvf1D<dist_t, attr_t>::CompassIvf1D(
    size_t d,
    size_t max_elements,
    size_t nlist,
    size_t nprobe,
    const float *xb
)
    : space_(d),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, std::numeric_limits<attr_t>::max()),
      btrees_(nlist, btree::btree_map<attr_t, labeltype>()),
      xb_(xb) {
  ivf_->nprobe = nprobe;
}

template <typename dist_t, typename attr_t>
int CompassIvf1D<dist_t, attr_t>::Add(const void *data_point, labeltype label, attr_t attr) {
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes
  // search_and_return_codes
  faiss::idx_t assigned_cluster;
  quantizer_.assign(1, (float *)data_point, &assigned_cluster, 1);
  btrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
void CompassIvf1D<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // assigned_clusters = new faiss::idx_t[n * 1];
  // ivfpq_.quantizer->assign(n, (float *)data, assigned_clusters, 1);
}

template <typename dist_t, typename attr_t>
int CompassIvf1D<dist_t, attr_t>::AddIvfPoints(
    size_t n,
    const void *data,
    labeltype *labels,
    const vector<attr_t> &attrs
) {
  ivf_->add(n, (float *)data);  // add_sa_codes
  auto assigned_clusters = new faiss::idx_t[n * 1];
  ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attrs[i];
    btrees_[assigned_clusters[i]].insert(std::make_pair(attrs[i], labels[i]));
  }
  delete[] assigned_clusters;
  return n;
}

template <typename dist_t, typename attr_t>
void CompassIvf1D<dist_t, attr_t>::SaveIvf(fs::path path) {
  fs::create_directories(path.parent_path());
  faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
}

template <typename dist_t, typename attr_t>
void CompassIvf1D<dist_t, attr_t>::LoadIvf(fs::path path) {
  auto ivf_file = fopen(path.c_str(), "r");
  auto index = faiss::read_index(ivf_file);
  this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
}