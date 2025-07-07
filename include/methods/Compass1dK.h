#pragma once

#include "basis/Compass1d.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

// For clustering transplanted to FAISS's KMeans implementation.
template <typename dist_t, typename attr_t>
class Compass1dK : public Compass1d<dist_t, attr_t> {
 public:
  Compass1dK(size_t n, size_t d, size_t M, size_t efc, size_t nlist) : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist) {
    this->ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist);
  }

  void AssignPoints(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) override {
    if (distances == nullptr) {
      dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_)->quantizer->assign(n, (float *)data, assigned_clusters, k);
    } else {
      dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_)
          ->quantizer->search(n, (float *)data, k, distances, assigned_clusters);
    }
  }
};
