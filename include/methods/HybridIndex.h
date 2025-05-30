#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <boost/filesystem.hpp>
#include <utility>
#include <vector>
#include "hnswlib.h"
#include "methods/Pod.h"
#include "methods/ReentrantHNSW.h"

namespace fs = boost::filesystem;
using hnswlib::L2Space;
using hnswlib::labeltype;
using std::pair;
using std::vector;

// Trainable, savable and loadable index.
template <typename dist_t, typename attr_t>
class HybridIndex {
 protected:
  ReentrantHNSW<dist_t> hnsw_;
  faiss::Index *ivf_;
  faiss::idx_t *base_cluster_rank_;   //  to speed up index loading
  faiss::idx_t *query_cluster_rank_;  // pre-allocated for query
  dist_t *distances_;                  // pre-allocated for query
  int n_, d_, M_, efc_, nlist_;

  // _GetClusters(const dist_t* query, const int nq, const int nprobe);

 public:
  HybridIndex(size_t n, size_t d, size_t M, size_t efc, size_t nlist)
      : n_(n),
        d_(d),
        M_(M),
        efc_(efc),
        nlist_(nlist),
        hnsw_(new L2Space(d), n, M, efc),
        base_cluster_rank_(new faiss::idx_t[n]),
        query_cluster_rank_(new faiss::idx_t[10000 * 1000]),
        distances_(new dist_t[10000 * 1000]) {}

  virtual void TrainIvf(size_t n, const dist_t *data) { ivf_->train(n, (float *)data); }
  virtual void AddPointsToGraph(const size_t n, const dist_t *data, const labeltype *labels) {
    for (int i = 0; i < n; i++) {
      this->hnsw_.addPoint(data + i * this->d_, labels[i], -1);
    }
  }

  virtual void AddPointsToIvf(const size_t n, const dist_t *data, const labeltype *labels, const attr_t *attrs) = 0;

  virtual void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    hnsw_.saveIndex(path.string());
  }
  virtual void LoadGraph(fs::path path) { hnsw_.loadIndex(path.string(), new L2Space(d_)); }
  virtual void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(ivf_), path.c_str());
  }
  virtual void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    ivf_ = dynamic_cast<faiss::Index *>(index);
  }

  virtual void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < this->n_; i++) {
      out.write((char *)(this->base_cluster_rank_ + i), sizeof(faiss::idx_t));
    }
  }

  virtual void LoadRanking(fs::path path, attr_t *attrs) = 0;

  virtual vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const dist_t *query,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) = 0;
};