#pragma once

#include "basis/Compass1dX.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

template <typename dist_t, typename attr_t>
class Compass1dXK : public Compass1dX<dist_t, attr_t> {
 public:
  Compass1dXK(size_t n, size_t d, size_t dx, size_t M, size_t efc, size_t nlist)
      : Compass1dX<dist_t, attr_t>(n, d, dx, M, efc, nlist) {
    this->ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
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
