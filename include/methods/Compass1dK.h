#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <queue>
#include <utility>
#include <vector>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "methods/Compass1d.h"

using std::pair;
using std::priority_queue;
using std::vector;

// For clustering transplanted to FAISS's KMeans implementation.
template <typename dist_t, typename attr_t>
class Compass1dK : public Compass1d<dist_t, attr_t> {
 protected:
  faiss::IndexIVFFlat *ivf_flat_;

 public:
  Compass1dK(size_t n, size_t d, size_t M, size_t efc, size_t nlist)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist),
        ivf_flat_(new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist)) {}

  void AssignPoints(const size_t n, const dist_t *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    if (distances == nullptr) {
      ivf_flat_->quantizer->assign(n, (float *)data, assigned_clusters, k);
    } else {
      ivf_flat_->quantizer->search(n, (float *)data, k, distances, assigned_clusters);
    }
  }
};
