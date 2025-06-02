#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <queue>
#include <utility>
#include <vector>
#include "Compass1d.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/MetricType.h"
#include "faiss/VectorTransform.h"

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass1dPca : public Compass1d<dist_t, attr_t> {
 protected:
  faiss::IndexIVFFlat *xivf_;
  faiss::PCAMatrix *pca_;
  faiss::IndexPreTransform *pca_ivf_;

 public:
  Compass1dPca(size_t n, size_t d, size_t M, size_t efc, size_t nlist, size_t dout)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist),
        xivf_(new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dout), dout, nlist)),
        pca_(new faiss::PCAMatrix(d, dout)),
        pca_ivf_(new faiss::IndexPreTransform(pca_, xivf_)) {
    pca_ivf_->prepend_transform(new faiss::CenteringTransform(d));
    pca_->eigen_power = -0.5;
    pca_->max_points_per_d = 2000;
    this->ivf_ = dynamic_cast<faiss::Index *>(pca_ivf_);
  }

  void AssignPoints(const size_t n, const dist_t *data, const int k, faiss::idx_t *assigned_clusters, float *distances)
      override {
    auto xdata = pca_ivf_->apply_chain(n, (float *)data);
    xivf_->quantizer->assign(n, (float *)xdata, assigned_clusters, k);
    delete[] xdata;
  }
};
