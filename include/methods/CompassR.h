#include <fmt/core.h>
#include <omp.h>
#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/geometry.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <queue>
#include <utility>
#include <vector>
#include "Pod.h"
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "hnswlib/hnswlib.h"
#include "methods/ReentrantHNSW.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
namespace geo = boost::geometry;

using point = geo::model::point<float, 2, geo::cs::cartesian>;
using box = geo::model::box<point>;
using value = std::pair<point, labeltype>;
using rtree = geo::index::rtree<value, geo::index::quadratic<16>>;

using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassR {
 private:
  L2Space space_;
  ReentrantHNSW<dist_t> hnsw_;
  faiss::IndexFlatL2 quantizer_;
  faiss::IndexIVFFlat *ivf_;
  // faiss::IndexIVFPQ ivfpq_;

  HierarchicalNSW<dist_t> cgraph_;

  vector<vector<attr_t>> attrs_;
  vector<rtree> rtrees_;

  faiss::idx_t *ranked_clusters_;

 public:
  CompassR(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist);
  int AddPoint(const void *data_point, labeltype label, attr_t attr);
  int AddGraphPoint(const void *data_point, labeltype label);
  int AddIvfPoints(size_t n, const void *data, labeltype *labels, const vector<vector<attr_t>> &attrs);
  void TrainIvf(size_t n, const void *data);

  // For CompassR series.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const vector<attr_t> &l_bounds,
      const vector<attr_t> &u_bounds,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    WindowQuery<float> pred(l_bounds, u_bounds, &attrs_);
    point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
    box b(min_corner, max_corner);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;
      auto itr_beg = rtrees_[ranked_clusters[curr_ci]].qbegin(geo::index::covered_by(b));
      auto itr_end = rtrees_[ranked_clusters[curr_ci]].qend();

      int cnt = 0;
      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1) && -candidate_set.top().first > distances[curr_ci])) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                itr_beg = rtrees_[ranked_clusters[curr_ci]].qbegin(geo::index::covered_by(b));
                itr_end = rtrees_[ranked_clusters[curr_ci]].qend();
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            candidate_set.emplace(-dist, tableid);
            if (dist < upper_bound) {
              top_candidates.emplace(dist, tableid);
              if (top_candidates.size() > efs_) top_candidates.pop();
            } else {
              recycle_set.emplace(-dist, tableid);
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
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  // For CompassRR series.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV1(
      const void *query,
      const int nq,
      const int k,
      const vector<attr_t> &l_bounds,
      const vector<attr_t> &u_bounds,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics,
      faiss::idx_t *ranked_clusters,
      float *distances
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;
    this->ivf_->quantizer->search(nq, (float *)query, nprobe, distances, ranked_clusters);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    WindowQuery<float> pred(l_bounds, u_bounds, &attrs_);
    point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
    box b(min_corner, max_corner);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(hnsw_.cur_element_count, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = q * nprobe;
      auto itr_beg = rtrees_[ranked_clusters[curr_ci]].qbegin(geo::index::covered_by(b));
      auto itr_end = rtrees_[ranked_clusters[curr_ci]].qend();

      int cnt = 0;
      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1) && -candidate_set.top().first > distances[curr_ci])) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                itr_beg = rtrees_[ranked_clusters[curr_ci]].qbegin(geo::index::covered_by(b));
                itr_end = rtrees_[ranked_clusters[curr_ci]].qend();
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          metrics[q].nround++;
          int cnt = hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            recycle_set.pop();
            cnt--;
          }
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
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  // For CompassRCg series.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnnV3(
      const void *query,
      const int nq,
      const int k,
      const vector<attr_t> &l_bounds,
      const vector<attr_t> &u_bounds,
      const int efs,
      const int nrel,
      const int nthread,
      vector<Metric> &metrics
  ) {
    auto efs_ = std::max(k, efs);
    hnsw_.setEf(efs_);
    int nprobe = ivf_->nlist / 20;

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));

    WindowQuery<float> pred(l_bounds, u_bounds, &attrs_);
    point min_corner(l_bounds[0], l_bounds[1]), max_corner(u_bounds[0], u_bounds[1]);
    box b(min_corner, max_corner);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;
      vector<pair<float, labeltype>> clusters = cgraph_.searchKnnCloserFirst((float *)(query) + q * ivf_->d, nprobe);

      vector<bool> visited(hnsw_.cur_element_count, false);

      metrics[q].nround = 0;
      metrics[q].ncomp = 0;

      int curr_ci = 0;
      auto itr_beg = rtrees_[clusters[curr_ci].second].qbegin(geo::index::covered_by(b));
      auto itr_end = rtrees_[clusters[curr_ci].second].qend();

      int cnt = 0;
      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe && -candidate_set.top().first > clusters[curr_ci].first)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= nprobe)
                break;
              else {
                itr_beg = rtrees_[clusters[curr_ci].second].qbegin(geo::index::covered_by(b));
                itr_end = rtrees_[clusters[curr_ci].second].qend();
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;
            visited[tableid] = true;

            auto vect = hnsw_.getDataByInternalId(tableid);
            auto dist = hnsw_.fstdistfunc_((float *)query + q * ivf_->d, vect, hnsw_.dist_func_param_);
            metrics[q].ncomp++;
            metrics[q].is_ivf_ppsl[tableid] = true;
            crel++;

            auto upper_bound = top_candidates.empty() ? std::numeric_limits<dist_t>::max() : top_candidates.top().first;
            candidate_set.emplace(-dist, tableid);
            if (dist < upper_bound) {
              top_candidates.emplace(dist, tableid);
              if (top_candidates.size() > efs_) top_candidates.pop();
            } else {
              recycle_set.emplace(-dist, tableid);
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
        if ((top_candidates.size() >= efs_) || curr_ci >= nprobe) {
          break;
        }
      }

      metrics[q].ncluster = curr_ci;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      metrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }

  void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->hnsw_.saveIndex(path.string());
  }
  void LoadGraph(fs::path path) { this->hnsw_.loadIndex(path.string(), &this->space_); }

  void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(this->ivf_), path.c_str());
  }
  void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    this->ivf_ = dynamic_cast<faiss::IndexIVFFlat *>(index);
  }

  void LoadRanking(fs::path path, const vector<vector<attr_t>> &attrs) {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      attrs_[i] = attrs[i];
      point p(attrs[i][0], attrs[i][1]);
      rtrees_[assigned_cluster].insert(std::make_pair(p, (labeltype)i));
    }
  }
  void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < hnsw_.max_elements_; i++) {
      out.write((char *)(ranked_clusters_ + i), sizeof(faiss::idx_t));
    }
  }

  void BuildClusterGraph() {
    auto centroids = ((faiss::IndexFlatL2 *)this->ivf_->quantizer)->get_xb();
    for (int i = 0; i < ivf_->nlist; i++) {
      this->cgraph_.addPoint(centroids + i * ivf_->d, i);
    }
  }
  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cgraph_.saveIndex(path.string());
  }
  void LoadClusterGraph(fs::path path) { this->cgraph_.loadIndex(path.string(), &this->space_); }
};

template <typename dist_t, typename attr_t>
CompassR<dist_t, attr_t>::CompassR(size_t d, size_t M, size_t efc, size_t max_elements, size_t nlist)
    : space_(d),
      hnsw_(&space_, max_elements, M, efc),
      quantizer_(d),
      ivf_(new faiss::IndexIVFFlat(&quantizer_, d, nlist)),
      attrs_(max_elements, vector<attr_t>()),
      rtrees_(nlist, rtree()),
      cgraph_(&space_, nlist, 8, 200) {
  ivf_->nprobe = nlist;
}

template <typename dist_t, typename attr_t>
int CompassR<dist_t, attr_t>::AddPoint(const void *data_point, labeltype label, attr_t attr) {
  hnsw_.addPoint(data_point, label, -1);
  attrs_[label] = attr;
  ivf_->add(1, (float *)data_point);  // add_sa_codes

  faiss::idx_t assigned_cluster;
  quantizer_.assign(1, (float *)data_point, &assigned_cluster, 1);
  rtrees_[assigned_cluster].insert(std::make_pair(attr, label));
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassR<dist_t, attr_t>::AddGraphPoint(const void *data_point, labeltype label) {
  hnsw_.addPoint(data_point, label, -1);
  return 1;
}

template <typename dist_t, typename attr_t>
int CompassR<dist_t, attr_t>::AddIvfPoints(
    size_t n,
    const void *data,
    labeltype *labels,
    const vector<vector<attr_t>> &attrs
) {
  ranked_clusters_ = new faiss::idx_t[n];
  ivf_->quantizer->assign(n, (float *)data, ranked_clusters_);
  for (int i = 0; i < n; i++) {
    attrs_[labels[i]] = attrs[i];
    point p(attrs[i][0], attrs[i][1]);
    rtrees_[ranked_clusters_[i]].insert(std::make_pair(p, labels[i]));
  }
  return n;
}

template <typename dist_t, typename attr_t>
void CompassR<dist_t, attr_t>::TrainIvf(size_t n, const void *data) {
  ivf_->train(n, (float *)data);
  // ivf_->add(n, (float *)data);
  // ivfpq_.train(n, (float *)data);
  // auto assigned_clusters = new faiss::idx_t[n * 1];
  // ivf_->quantizer->assign(n, (float *)data, assigned_clusters, 1);
}
