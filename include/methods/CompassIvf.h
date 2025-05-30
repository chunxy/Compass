#pragma once

#include <fmt/core.h>
#include <boost/filesystem.hpp>
#include <boost/geometry.hpp>
#include <cstddef>
#include <cstdlib>
#include <queue>
#include <utility>
#include <vector>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/Pod.h"

using hnswlib::L2Space;
using hnswlib::labeltype;

namespace fs = boost::filesystem;

namespace geo = boost::geometry;
using point = geo::model::point<float, 2, geo::cs::cartesian>;
using box = geo::model::box<point>;
using value = std::pair<point, labeltype>;
using rtree = geo::index::rtree<value, geo::index::quadratic<16>>;

using std::pair;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassIvf {
 private:
  L2Space space_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  vector<vector<attr_t>> attrs_;
  vector<rtree> rtrees_;

  const float *xb_;
  faiss::idx_t *ranked_clusters_;

 public:
  CompassIvf(size_t d, size_t max_elements, size_t nlist, const float *xb);
  int AddPointsToIvf(size_t n, const void *data, labeltype *labels, const vector<vector<attr_t>> &attrs);
  void TrainIvf(size_t n, const void *data);
  vector<vector<pair<float, labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const vector<attr_t> &l_bounds,
      const vector<attr_t> &u_bounds,
      const int nprobe,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters
  ) {
    ivf_->quantizer->assign(nq, (float *)query, ranked_clusters, nprobe);
    // auto &dm = ivf_->direct_map;

    vector<vector<pair<float, labeltype>>> result(nq, vector<pair<float, labeltype>>(k));

    // point min_corner(l_bounds[0], u_bounds[0]), max_corner(l_bounds[1], u_bounds[1]);
    point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
    box b(min_corner, max_corner);

    for (int q = 0; q < nq; q++) {
      std::priority_queue<pair<float, labeltype>> top_candidates;
      int i = 0;
      for (i = 0; i < nprobe; i++) {
        auto cluster = ranked_clusters[q * nprobe + i];
        if (cluster == -1) break;
        auto rel_beg = rtrees_[cluster].qbegin(geo::index::covered_by(b));
        auto rel_end = rtrees_[cluster].qend();
        while (rel_beg != rel_end) {
          auto j = (*rel_beg).second;
          metrics[q].is_ivf_ppsl[j] = true;
          const dist_t *vect = xb_ + j * quantizer_.d;
          auto dist = space_.get_dist_func()((dist_t *)query + q * ivf_->d, vect, space_.get_dist_func_param());
          metrics[q].ncomp++;
          top_candidates.emplace(dist, j);
          rel_beg++;
        }
      }
      metrics[q].ncluster = i;

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

  void LoadRanking(fs::path path, const vector<vector<attr_t>> &attrs) {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < attrs_.size(); i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      attrs_[i] = attrs[i];
      point p(attrs[i][0], attrs[i][1]);
      rtrees_[assigned_cluster].insert(std::make_pair(p, (labeltype)i));
    }
  }

  void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < attrs_.size(); i++) {
      out.write((char *)(ranked_clusters_ + i), sizeof(faiss::idx_t));
    }
  }
};

template <typename dist_t, typename attr_t>
CompassIvf<dist_t, attr_t>::CompassIvf(size_t d, size_t max_elements, size_t nlist, const float *xb)
    : space_(d),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, vector<attr_t>()),
      rtrees_(nlist, rtree()),
      xb_(xb) {}

template <typename dist_t, typename attr_t>
int CompassIvf<dist_t, attr_t>::AddPointsToIvf(
    size_t n,
    const void *data,
    labeltype *labels,
    const vector<vector<attr_t>> &attrs
) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  ranked_clusters_ = new faiss::idx_t[n];
  ivf_->quantizer->assign(n, (float *)data, ranked_clusters_);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attrs[i];
    point p(attrs[i][0], attrs[i][1]);
    rtrees_[ranked_clusters_[i]].insert(std::make_pair(p, labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassIvf<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // assigned_clusters = new faiss::idx_t[n * 1];
  // ivfpq_.quantizer->assign(n, (float *)data, assigned_clusters, 1);
}

template <typename dist_t, typename attr_t>
void CompassIvf<dist_t, attr_t>::SaveIvf(fs::path path) {
  fs::create_directories(path.parent_path());
  faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
}

template <typename dist_t, typename attr_t>
void CompassIvf<dist_t, attr_t>::LoadIvf(fs::path path) {
  auto ivf_file = fopen(path.c_str(), "r");
  auto index = faiss::read_index(ivf_file);
  this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
}