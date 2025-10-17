#include <fmt/core.h>
#include <cstdlib>
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "methods/basis/CompassPost.h"

template <typename dist_t, typename attr_t>
class CompassPostK : public CompassPost<dist_t, attr_t> {
 public:
  CompassPostK(
      size_t n,
      size_t d,
      size_t da,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t initial_efs,
      size_t delta_efs
  )
      : CompassPost<dist_t, attr_t>(n, d, da, M, efc, nlist, M_cg, batch_k, initial_efs, delta_efs) {}

  void LoadRanking(fs::path path, attr_t *attrs) override {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < this->n_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      array<attr_t, 4> arr{0, 0, 0, 0};
      for (int j = 0; j < this->da_; j++) {
        arr[j] = attrs[i * this->da_ + j];
        this->mbtrees_[assigned_cluster * this->da_ + j].insert(
            frozenca::BTreePair<attr_t, labeltype>(std::move(attrs[i * this->da_ + j]), (labeltype)i)
        );
      }
      this->btrees_[assigned_cluster].insert(frozenca::BTreePair<attr_t, pair<labeltype, array<attr_t, 4>>>(
          std::move(attrs[i * this->da_]), {(labeltype)i, arr}
      ));
    }
    int sz = 0, msz = 0;
    for (int i = 0; i < this->btrees_.size(); i++) {
      sz += this->btrees_[i].size();
    }
    for (int i = 0; i < this->mbtrees_.size(); i++) {
      msz += this->mbtrees_[i].size();
    }
    fmt::print("B-tree size: {}\n", sz);
    fmt::print("MB-tree size: {}\n", msz);
    if (sz != this->n_ || msz % this->n_ != 0) {
      exit(-1);
    }
  }

  void LoadRanking(fs::path path, attr_t *attrs, int which) {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < this->n_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      array<attr_t, 4> arr{0, 0, 0, 0};
      for (int j = 0; j < this->da_; j++) {
        arr[j] = attrs[i * this->da_ + j];
        this->mbtrees_[assigned_cluster * this->da_ + j].insert(
            frozenca::BTreePair<attr_t, labeltype>(std::move(attrs[i * this->da_ + j]), (labeltype)i)
        );
      }
      this->btrees_[assigned_cluster].insert(frozenca::BTreePair<attr_t, pair<labeltype, array<attr_t, 4>>>(
          std::move(attrs[i * this->da_ + which]), {(labeltype)i, arr}
      ));
    }
    int sz = 0, msz = 0;
    for (int i = 0; i < this->btrees_.size(); i++) {
      sz += this->btrees_[i].size();
    }
    for (int i = 0; i < this->mbtrees_.size(); i++) {
      msz += this->mbtrees_[i].size();
    }
    fmt::print("B-tree size: {}\n", sz);
    fmt::print("MB-tree size: {}\n", msz);
    if (sz != this->n_ || msz % this->n_ != 0) {
      exit(-1);
    }
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
      this->cg_.hnsw_->addPoint(centroids + i * ivf_flat->d, i);
    }
  }
};