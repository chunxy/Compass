#pragma once

#include "basis/CompassCg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "hnswlib/hnswlib.h"

template <typename dist_t, typename attr_t>
class CompassKCg : public CompassCg<dist_t, attr_t> {
 public:
  CompassKCg(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist, size_t M_cg)
      : CompassCg<dist_t, attr_t>(n, d, da, M, efc, nlist, M_cg) {
    this->ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist);
  }

  void AssignPoints(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) override {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    if (distances) {
      ivf_flat->quantizer->search(n, (float *)data, k, distances, assigned_clusters);
    } else {
      ivf_flat->quantizer->assign(n, (float *)data, assigned_clusters, k);
    }
  };

  void BuildClusterGraph() override {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->cgraph_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }
};
