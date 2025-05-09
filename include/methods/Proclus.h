#pragma once
#include <cstdint>
#include "faiss/MetricType.h"
#include "space_l1.h"

struct Proclus {
  float *medoids;
  float *mask;  // Changed from int32_t* to float*
  int32_t *assignment;
  int nclusters;
  int nb;
  int d;
  hnswlib::L1MaskedSpace space;
  hnswlib::L2Space l2space;

  Proclus(int nclusters, int d) : nclusters(nclusters), d(d), space(d), l2space(d) {
    medoids = new float[nclusters * d];
    mask = new float[nclusters * d];  // Changed from int32_t* to float*
    memset(mask, 0, nclusters * d * sizeof(float));
    assignment = nullptr;
  }

  Proclus(int nclusters, int nb, int d) : nclusters(nclusters), nb(nb), d(d), space(d), l2space(d) {
    medoids = new float[nclusters * d];
    mask = new float[nclusters * d];  // Changed from int32_t* to float*
    memset(mask, 0, nclusters * d * sizeof(float));
    assignment = new int32_t[nb];
  }

  // write a function read in medoids and mask from float*
  void read(float *medoids, int32_t *subspaces, int32_t *assignment) {
    memcpy(this->medoids, medoids, nclusters * d * sizeof(float));
    memset(this->mask, 0, nclusters * d * sizeof(float));
    for (int i = 0; i < nclusters; i++) {
      for (int j = 0; j < d; j++) {
        if (subspaces[i * d + j] == -1) break;
        this->mask[i * d + subspaces[i * d + j]] = 1;
      }
    }
    memcpy(this->assignment, assignment, nb * sizeof(int32_t));
  }

  void read_subspaces(std::string path) {
    std::ifstream in(path);
    for (int i = 0; i < nclusters; i++) {
      for (int j = 0; j < d; j++) {
        int32_t value;
        in.read((char *)&value, sizeof(int32_t));
        if (value == -1) continue;  // Skip the padded -1
        mask[i * d + value] = 1;
      }
    }
  }

  void search_rerank(int n, float *x, faiss::idx_t *labels, float *distances, int k = 1) {
    std::fill(labels, labels + n, -1);
    std::fill(distances, distances + n * k, std::numeric_limits<float>::max());
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      std::priority_queue<std::pair<float, int>> max_heap;
      for (int j = 0; j < nclusters; j++) {
        float dist = space.get_dist_func()(x + i * d, medoids + j * d, space.get_dist_func_param(), mask + j * d);
        max_heap.emplace(dist, j);
        if (max_heap.size() > k) {
          max_heap.pop();
        }
      }
      std::priority_queue<std::pair<float, int>> max_heap_l2;
      while (!max_heap.empty()) {
        auto top = max_heap.top();
        float dist = l2space.get_dist_func()(x + i * d, medoids + top.second * d, l2space.get_dist_func_param());
        max_heap_l2.emplace(-dist, top.second);
        max_heap.pop();
      }
      int j = 0;
      while (!max_heap_l2.empty()) {
        auto top = max_heap_l2.top();
        labels[i * k + j] = top.second;
        distances[i * k + j] = -top.first;
        j++;
        max_heap_l2.pop();
      }
    }
  }

  void search(int n, float *x, faiss::idx_t *labels, int k = 1) {
    std::fill(labels, labels + n, -1);
    // #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      std::priority_queue<std::pair<float, int>> max_heap;
      for (int j = 0; j < nclusters; j++) {
        float dist = space.get_dist_func()(x + i * d, medoids + j * d, space.get_dist_func_param(), mask + j * d);
        max_heap.emplace(dist, j);
        if (max_heap.size() > k) {
          max_heap.pop();
        }
      }
      int j = 0;
      while (!max_heap.empty()) {
        auto top = max_heap.top();
        labels[i * k + j] = top.second;
        j++;
        max_heap.pop();
      }
    }
  }
};