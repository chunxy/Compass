#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <boost/filesystem.hpp>
#include <utility>
#include <vector>
#include "hnswlib.h"
#include "methods/Pod.h"

namespace fs = boost::filesystem;
using std::pair;
using std::vector;

template <typename dist_t, typename attr_t>
class HybridIndex {
 public:
  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }
  void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
  }
  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }
  void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
  }

  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int nthread,
      Metric &metric
  );
};