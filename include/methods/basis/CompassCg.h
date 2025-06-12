#pragma once

#include "Compass.h"
#include "faiss/MetricType.h"

template <typename dist_t, typename attr_t>
class CompassCg : public Compass<dist_t, attr_t> {
 protected:
  HierarchicalNSW<dist_t> *cgraph_;

 public:
  CompassCg(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist)
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist) {
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(d), nlist, 8, 200);
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
      auto clusters = cgraph_->searchKnnCloserFirst((float *)&data[i * this->d_], k);
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

  virtual void BuildClusterGraph() = 0;

  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cgraph_->saveIndex(path.string());
  }

  virtual void LoadClusterGraph(fs::path path) {
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(this->d_), path.string());
  }
};