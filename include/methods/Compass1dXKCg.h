#pragma once

#include "basis/Compass1dXCg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

template <typename dist_t, typename attr_t>
class Compass1dXKCg : public Compass1dXCg<dist_t, attr_t> {
 protected:
  HierarchicalNSW<dist_t> cgraph_;

 public:
  Compass1dXKCg(size_t n, size_t d, size_t dx, size_t M, size_t efc, size_t nlist, size_t M_cg)
      : Compass1dXCg<dist_t, attr_t>(n, d, dx, M, efc, nlist, M_cg) {
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

  void BuildClusterGraph() {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->cgraph_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }
};
