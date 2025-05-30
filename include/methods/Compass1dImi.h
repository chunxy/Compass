#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <queue>
#include <utility>
#include <vector>
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPQ.h"
#include "faiss/MetricType.h"
#include "methods/Compass1d.h"

using std::pair;
using std::priority_queue;
using std::vector;

// For clustering transplanted to FAISS's KMeans implementation.
template <typename dist_t, typename attr_t>
class Compass1dImi : public Compass1d<dist_t, attr_t> {
 protected:
  faiss::IndexIVFFlat *ivf_flat_;

 public:
  Compass1dImi(size_t n, size_t d, size_t M, size_t efc, size_t nsub, size_t nbits)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, 1 << (nbits * nsub)),
        ivf_flat_(new faiss::IndexIVFFlat(new faiss::MultiIndexQuantizer(d, nsub, nbits), d, 1 << (nbits * nsub))) {
    ivf_flat_->quantizer_trains_alone = true;
  }

  void AssignPoints(const size_t n, const dist_t *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    if (distances == nullptr) {
      ivf_flat_->quantizer->assign(n, (float *)data, assigned_clusters, k);
    } else {
      ivf_flat_->quantizer->search(n, (float *)data, k, distances, assigned_clusters);
    }
  }
};
