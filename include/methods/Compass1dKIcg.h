#pragma once

#include <cstddef>
#include "basis/Compass1dIcg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"

template <typename dist_t, typename attr_t>
class Compass1dKIcg : public Compass1dIcg<dist_t, attr_t> {
 public:
  Compass1dKIcg(
      size_t n,
      size_t d,
      SpaceInterface<dist_t> *s,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t delta_efs
  )
      : Compass1dIcg<dist_t, attr_t>(n, d, s, M, efc, nlist, M_cg, batch_k, delta_efs) {
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

  void BuildClusterGraph() override {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->isearch_->hnsw_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }
};
