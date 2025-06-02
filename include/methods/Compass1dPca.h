#pragma once

#include "Compass1d.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/MetricType.h"
#include "faiss/VectorTransform.h"

template <typename dist_t, typename attr_t>
class Compass1dPca : public Compass1d<dist_t, attr_t> {
 public:
  Compass1dPca(size_t n, size_t d, size_t M, size_t efc, size_t nlist, size_t dx)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist) {
    auto xivf = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
    auto pca = new faiss::PCAMatrix(d, dx);
    // pca->eigen_power = -0.5;
    // pca->max_points_per_d = 2000;
    auto pca_ivf = new faiss::IndexPreTransform(pca, xivf);
    pca_ivf->prepend_transform(new faiss::CenteringTransform(d));
    this->ivf_ = pca_ivf;
  }

  void AssignPoints(const size_t n, const dist_t *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto xdata = ivf_trans->apply_chain(n, (float *)data);
    dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index)->quantizer->assign(n, xdata, assigned_clusters, k);
    delete[] xdata;
  }
};
