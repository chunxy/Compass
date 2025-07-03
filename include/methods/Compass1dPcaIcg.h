#pragma once

#include "basis/Compass1dXIcg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"

template <typename dist_t, typename attr_t>
class Compass1dPcaIcg : public Compass1dXIcg<dist_t, attr_t> {
 protected:
  IterativeSearchState<dist_t> *Open(const dist_t *query, int idx, int nprobe) override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto xquery = ivf_trans->apply_chain(1, query + idx * this->d_);
    auto ret = this->isearch_->Open(xquery, nprobe);
    delete[] xquery;
    return ret;
  }

 public:
  Compass1dPcaIcg(
      size_t n,
      size_t d,
      size_t dx,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t delta_efs
  )
      : Compass1dXIcg<dist_t, attr_t>(n, d, dx, M, efc, nlist, M_cg, batch_k, delta_efs) {
    auto xivf = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
    auto pca = new faiss::PCAMatrix(d, dx);
    // pca->eigen_power = -0.5;
    // pca->max_points_per_d = 2000;
    auto pca_ivf = new faiss::IndexPreTransform(pca, xivf);
    pca_ivf->prepend_transform(new faiss::CenteringTransform(d));
    this->ivf_ = pca_ivf;
  }

  void AssignPoints(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto xdata = ivf_trans->apply_chain(n, (float *)data);
    dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index)->quantizer->assign(n, xdata, assigned_clusters, k);
    delete[] xdata;
  }

  void BuildClusterGraph() override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->isearch_->hnsw_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }
};
