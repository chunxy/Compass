#pragma once

#include "basis/CompassX.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/MetricType.h"
#include "faiss/VectorTransform.h"

// CompassPca* is a bit different from CompassX* in that
// it can transform the data on its own instead of
// relying on the arguments passed from the caller.

// But this transformation may be time-consuming.
// So still consider passing the trasformed data
// from the caller in future.
// To do so, rewrite the current AssignPoints and SearchClusters
// so that they directly operate on the transformed data; and
// pass the transformed data from the caller.
template <typename dist_t, typename attr_t>
class CompassPca : public CompassX<dist_t, attr_t> {
 public:
  CompassPca(size_t n, size_t d, size_t dx, size_t da, size_t M, size_t efc, size_t nlist)
      : CompassX<dist_t, attr_t>(n, d, dx, da, M, efc, nlist) {
    auto xivf = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
    auto pca = new faiss::PCAMatrix(d, dx);
    // pca->eigen_power = -0.5;
    // pca->max_points_per_d = 2000;
    auto pca_ivf = new faiss::IndexPreTransform(pca, xivf);
    pca_ivf->prepend_transform(new faiss::CenteringTransform(d));
    this->ivf_ = pca_ivf;
  }

  void AssignPoints(const size_t n, const void *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto xdata = ivf_trans->apply_chain(n, (float *)data);
    dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index)->quantizer->assign(n, xdata, assigned_clusters, k);
    delete[] xdata;
  }
};
