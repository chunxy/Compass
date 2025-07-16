#pragma once

#include <cstddef>
#include "basis/Compass1dIcg.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexScalarQuantizer.h"
#include "fmt/core.h"

template <typename dist_t, typename attr_t>
class Compass1dKQicg : public Compass1dIcg<dist_t, attr_t, int> {
 private:
  faiss::IndexScalarQuantizer *sq_;
  uint8_t *query_code_;

 protected:
  IterativeSearchState<int> Open(const void *query, int idx, int nprobe) override {
    // void *target = ((char *)query) + this->hnsw_.data_size_ * idx;
    // sq_->sa_encode(1, (float *)target, query_code_);
    // return this->isearch_->Open(query_code_, nprobe);
    if (sq_->code_size != this->isearch_->hnsw_->data_size_) {
      fmt::print("Scalar quantizer code size does not match HNSW data size.");
      exit(-1);
    }
    return this->isearch_->Open((char *)query + idx * sq_->code_size, nprobe);
  }

  const void *icg_transform(const void *query, int nq) override {
    sq_->sa_encode(nq, (float *)query, query_code_);
    return query_code_;
  }

 public:
  Compass1dKQicg(
      size_t n,
      size_t d,
      SpaceInterface<int> *s,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t initial_efs,
      size_t delta_efs
  )
      : Compass1dIcg<dist_t, attr_t, int>(n, d, s, M, efc, nlist, M_cg, batch_k, initial_efs, delta_efs) {
    this->ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist);
    sq_ = new faiss::IndexScalarQuantizer(d, faiss::ScalarQuantizer::QT_8bit_uniform);
    query_code_ = new uint8_t[1000 * sq_->code_size];
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

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->isearch_->hnsw_->saveIndex(path.string());
  }

  void LoadClusterGraph(fs::path path) override {
    this->isearch_->hnsw_->loadIndex(path.string(), new L2SpaceB(this->d_));
  }

  void BuildClusterGraph() override {
    auto ivf_flat = dynamic_cast<faiss::IndexIVFFlat *>(this->ivf_);
    auto centroids = ((faiss::IndexFlatL2 *)ivf_flat->quantizer)->get_xb();

    uint8_t *codes = new uint8_t[ivf_flat->nlist * sq_->code_size];
    sq_->train(ivf_flat->nlist, centroids);
    sq_->sa_encode(ivf_flat->nlist, centroids, codes);
    for (int i = 0; i < ivf_flat->nlist; i++) {
      this->isearch_->hnsw_->addPoint(codes + i * sq_->code_size, i);
    }
  }

  void SaveScalarQuantizer(fs::path path) { faiss::write_index((faiss::Index *)this->sq_, path.c_str()); }

  void LoadScalarQuantizer(fs::path path) {
    auto index = faiss::read_index(path.c_str());
    if (sq_) delete sq_;
    sq_ = dynamic_cast<faiss::IndexScalarQuantizer *>(index);
  }
};
