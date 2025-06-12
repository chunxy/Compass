#pragma once

#include "CompassCg.h"

template <typename dist_t, typename attr_t>
class CompassXCg : public CompassCg<dist_t, attr_t> {
 protected:
  int dx_;

 public:
  CompassXCg(size_t n, size_t d, size_t dx, size_t da, size_t M, size_t efc, size_t nlist)
      : CompassCg<dist_t, attr_t>(n, d, da, M, efc, nlist), dx_(dx) {
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(dx), nlist, 8, 200);
  }

  void SearchClusters(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) override {
    for (int i = 0; i < n; i++) {
      auto clusters = this->cgraph_->searchKnnCloserFirst((float *)&data[i * dx_], k);
      for (int j = 0; j < k; j++) {
        assigned_clusters[i * k + j] = clusters[j].second;
      }
      if (distances) {
        for (int j = 0; j < k; j++) {
          distances[i * k + j] = clusters[j].first;
        }
      }
    }
  }

  void LoadClusterGraph(fs::path path) override {
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(dx_), path.string());
  }
};