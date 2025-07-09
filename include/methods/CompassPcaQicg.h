#pragma once

#include <cstddef>
#include "basis/CompassXIcg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/IndexScalarQuantizer.h"

template <typename dist_t, typename attr_t>
class CompassPcaQicg : public CompassXIcg<dist_t, attr_t, int> {
 private:
  faiss::IndexScalarQuantizer *sq_;
  uint8_t *query_code_;

 protected:
  IterativeSearchState<int> Open(const void *query, int idx, int nprobe) override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    const void *target = ((char *)query) + idx * this->hnsw_.data_size_;
    auto xquery = ivf_trans->apply_chain(1, (float *)target);
    sq_->sa_encode(1, (float *)xquery, query_code_);
    // delete[] xquery;
    return this->isearch_->Open(query_code_, nprobe);
  }

 public:
  CompassPcaQicg(
      size_t n,
      size_t d,
      size_t dx,
      SpaceInterface<int> *s,
      size_t da,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t delta_efs
  )
      : CompassXIcg<dist_t, attr_t, int>(n, d, dx, s, da, M, efc, nlist, M_cg, batch_k, delta_efs) {
    auto xivf = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(dx), dx, nlist);
    auto pca = new faiss::PCAMatrix(d, dx);
    // pca->eigen_power = -0.5;
    // pca->max_points_per_d = 2000;
    auto pca_ivf = new faiss::IndexPreTransform(pca, xivf);
    pca_ivf->prepend_transform(new faiss::CenteringTransform(d));
    this->ivf_ = pca_ivf;
    sq_ = new faiss::IndexScalarQuantizer(dx, faiss::ScalarQuantizer::QT_8bit_uniform);
    query_code_ = new uint8_t[sq_->code_size];
  }

  void AssignPoints(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto xdata = ivf_trans->apply_chain(n, (float *)data);
    dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index)->quantizer->assign(n, xdata, assigned_clusters, k);
    delete[] xdata;
  }

  void LoadClusterGraph(fs::path path) override {
    this->isearch_->hnsw_->loadIndex(path.string(), new L2SpaceB(this->dx_));
  }

  void BuildClusterGraph() override {
    auto ivf_trans = dynamic_cast<faiss::IndexPreTransform *>(this->ivf_);
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(ivf_trans->index);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->isearch_->hnsw_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }

  void SaveScalarQuantizer(fs::path path) { faiss::write_index((faiss::Index *)this->sq_, path.c_str()); }

  void LoadScalarQuantizer(fs::path path) {
    auto index = faiss::read_index(path.c_str());
    if (sq_) delete sq_;
    sq_ = dynamic_cast<faiss::IndexScalarQuantizer *>(index);
  }
};
