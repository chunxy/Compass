#include <chrono>
#include <cstddef>
#include "Compass1dCg.h"
#include "hnswlib/hnswalg.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass1dXCg : public Compass1dCg<dist_t, attr_t> {
 protected:
  int dx_;

 public:
  Compass1dXCg(size_t n, size_t d, size_t dx, size_t M, size_t efc, size_t nlist, size_t M_cg)
      : Compass1dCg<dist_t, attr_t>(n, d, M, efc, nlist, M_cg), dx_(dx) {
    if (this->cgraph_) delete this->cgraph_;
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(dx), nlist, M_cg, 200);
  }

  void SearchClusters(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) override {
    auto search_beg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
      auto clusters = this->cgraph_->searchKnnCloserFirst((char *)data + i * this->cgraph_->data_size_, k);
      for (int j = 0; j < k; j++) {
        assigned_clusters[i * k + j] = clusters[j].second;
      }
      if (distances) {
        for (int j = 0; j < k; j++) {
          distances[i * k + j] = clusters[j].first;
        }
      }
    }
    auto search_end = std::chrono::high_resolution_clock::now();
    bm.cluster_search_time = std::chrono::duration_cast<std::chrono::nanoseconds>(search_end - search_beg).count();
  }

  void LoadClusterGraph(fs::path path) override { this->cgraph_->loadIndex(path.string(), new L2Space(dx_)); }
};