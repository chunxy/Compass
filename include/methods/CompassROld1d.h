#pragma once

#include <fmt/core.h>
#include <omp.h>
#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "Pod.h"
#include "btree_map.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/ReentrantHNSW.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
using coroutine_t = boost::coroutines2::coroutine<int>;

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassROld1d {
 public:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  vector<attr_t> attrs_;
  vector<btree::btree_map<attr_t, labeltype>> btrees_;

  faiss::idx_t *ranked_clusters_;

  // config
  size_t nrel_;

 public:
  CompassROld1d(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist, size_t nrel, size_t nbits);
  int AddPoint(const void *data_point, labeltype label, attr_t attr);
  int AddGraphPoint(const void *data_point, labeltype label);
  int AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attrs);
  void TrainIvf(size_t n, const void *data);

  vector<priority_queue<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t &l_bound,
      const attr_t &u_bound,
      const int efs,
      const int min_comp,
      const int nthread,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    // int nprobe = ivf_->nlist;
    int nprobe = 100;
    // auto centroids = quantizer_.get_xb();
    // auto dist_func = quantizer_.get_distance_computer();
    // auto ranked_clusters = new faiss::idx_t[nq * nprobe];
    // auto distances = new float[nq * nprobe];
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq, priority_queue<pair<dist_t, labeltype>>());

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      RangeQuery<float> pred(l_bound, u_bound, &attrs_);
      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      {
        tableint currObj = hnsw_.enterpoint_node_;
        dist_t curdist = hnsw_.fstdistfunc_(
            (float *)query + q * ivf_->d, hnsw_.getDataByInternalId(hnsw_.enterpoint_node_), hnsw_.dist_func_param_
        );

        for (int level = hnsw_.maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)hnsw_.get_linklist(currObj, level);
            int size = hnsw_.getListCount(data);
            // metric_hops++;
            // metric_distance_computations += size;
            metrics[q].ncomp += size;

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > hnsw_.max_elements_) throw std::runtime_error("cand error");
              dist_t d = hnsw_.fstdistfunc_(
                  (float *)query + q * ivf_->d, hnsw_.getDataByInternalId(cand), hnsw_.dist_func_param_
              );

              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
        // ranked_clusters = ranked_clusters_ + currObj * nprobe;
        visited[currObj] = true;
        candidate_set.emplace(-curdist, currObj);
      }

      auto curr_ci = q * nprobe;
      auto itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
      auto itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);

      int cnt = 0;
      while (true) {
        int crel = 0;
        // if (candidate_set.empty() || distances[curr_ci] <
        // -candidate_set.top().first) {
        if (candidate_set.empty() ||
            (!top_candidates.empty() && -candidate_set.top().first > top_candidates.top().first)) {
          while (crel < nrel_) {
            if (itr_beg == itr_end) {
              if (++curr_ci == (q + 1) * nprobe)
                break;
              else {
                itr_beg = btrees_[ranked_clusters[curr_ci]].lower_bound(l_bound);
                itr_end = btrees_[ranked_clusters[curr_ci]].upper_bound(u_bound);
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            if (top_candidates.size() < efs || dist < upper_bound) {
              candidate_set.emplace(-dist, tableid);
              top_candidates.emplace(dist, tableid);
              metrics[q].is_ivf_ppsl[tableid] = true;
              if (top_candidates.size() > efs_) top_candidates.pop();
            }
          }
          metrics[q].nround++;
        }

        hnsw_.ReentrantSearchKnn(
            (float *)query + q * ivf_->d,
            k,
            -1,
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(metrics[q].ncomp),
            std::ref(metrics[q].is_graph_ppsl)
        );
        // if ((top_candidates.size() >= efs_ && min_comp - metrics[q].ncomp <
        // 0) ||
        //     curr_ci >= (q + 1) * nprobe) {
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      while (top_candidates.size() > k) top_candidates.pop();
      // size_t sz = top_candidates.size();
      // vector<std::pair<dist_t, labeltype>> result(sz);
      while (!top_candidates.empty()) {
        results[q].push(top_candidates.top());
        top_candidates.pop();
      }
    }

    return results;
  }

  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }

  void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
  }

  void LoadRanking(fs::path path, attr_t *attrs) {
    std::ifstream in(path.string());
    ranked_clusters_ = new faiss::idx_t[hnsw_.max_elements_ * ivf_->nlist];
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      for (int j = 0; j < ivf_->nlist; j++) {
        in.read((char *)(ranked_clusters_ + i * ivf_->nlist + j), sizeof(faiss::idx_t));
      }
    }
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      attrs_[i] = attrs[i];
      btrees_[ranked_clusters_[i * ivf_->nlist]].insert(std::make_pair(attrs[i], (labeltype)i));
    }
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }

  void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
  }

  void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      for (int j = 0; j < ivf_->nlist; j++) {
        out.write((char *)(ranked_clusters_ + i * ivf_->nlist + j), sizeof(faiss::idx_t));
      }
    }
  }
};

template <typename dist_t, typename attr_t>
CompassROld1d<dist_t, attr_t>::CompassROld1d(
    size_t d,
    size_t M,
    size_t efc,
    size_t max_elements,
    size_t nlist,
    size_t nrel,
    size_t nbits
)
    : space_(d),
      hnsw_(&space_, max_elements, M, efc),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, std::numeric_limits<attr_t>::max()),
      btrees_(nlist, btree::btree_map<attr_t, labeltype>()),
      nrel_(nrel) {
  ivf_->nprobe = nlist;
}

template <typename dist_t, typename attr_t>
int CompassROld1d<dist_t, attr_t>::AddPoint(const void *data_point, labeltype label, attr_t attr) {
  hnsw_.addPoint(data_point, label, -1);
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes

  faiss::idx_t assigned_cluster;
  quantizer_.assign(1, (float *)data_point, &assigned_cluster, 1);
  btrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassROld1d<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassROld1d<dist_t, attr_t>::AddIvfPoints(size_t n, const void *data, labeltype *labels, attr_t *attr) {
  // ivf_->add(n, (float *)data);  // add_sa_codes
  ranked_clusters_ = new faiss::idx_t[n * ivf_->nlist];
  ivf_->quantizer->assign(n, (float *)data, ranked_clusters_, ivf_->nlist);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attr[i];
    btrees_[ranked_clusters_[i * ivf_->nlist]].insert(std::make_pair(attr[i], labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassROld1d<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivf_->add(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // auto assigned_clusters = new faiss::idx_t[n * 1];
  // ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
}
