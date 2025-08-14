#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "methods/basis/Compass1dPost.h"

template <typename dist_t, typename attr_t>
class Compass1dPostK : public Compass1dPost<dist_t, attr_t> {
 public:
  Compass1dPostK(
      size_t n,
      size_t d,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t initial_efs,
      size_t delta_efs
  )
      : Compass1dPost<dist_t, attr_t>(n, d, M, efc, nlist, M_cg, batch_k, initial_efs, delta_efs) {}

  void LoadRanking(fs::path path, attr_t *attrs) override {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < this->n_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      this->btrees_[assigned_cluster].insert(std::make_pair(attrs[i], (labeltype)i));
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