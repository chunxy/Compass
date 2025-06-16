#pragma once

#include "Compass.h"
#include "faiss/MetricType.h"

template <typename dist_t, typename attr_t>
class CompassCg : public Compass<dist_t, attr_t> {
 protected:
  HierarchicalNSW<dist_t> *cgraph_;

 public:
  CompassCg(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist, size_t M_cg)
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist) {
    this->cgraph_ = new HierarchicalNSW<dist_t>(new L2Space(d), nlist, M_cg, 200);
  }

  void SearchClusters(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) override {
    int count_beg = this->cgraph_->metric_distance_computations;
    auto search_beg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
      auto clusters = cgraph_->searchKnnCloserFirst(&data[i * this->d_], k);
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
    int count_end = this->cgraph_->metric_distance_computations;
    bm.cluster_search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_beg).count();
    bm.cluster_search_ncomp = count_end - count_beg;
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