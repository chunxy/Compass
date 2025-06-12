#include <variant>
#include "HybridIndex.h"
#include "btree_map.h"
#include "utils/predicate.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass : public HybridIndex<dist_t, attr_t> {
 protected:
  vector<vector<btree::btree_map<attr_t, labeltype>>> btrees_;
  int da_;

  // Assign original/transformed points to clusters.
  // Called during index building.
  virtual void AssignPoints(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) = 0;

  // Potentially assign original/transformed points to clusters using cluster graph,
  // as well as profile the initial search process.
  virtual void SearchClusters(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) {
    auto assign_beg = std::chrono::high_resolution_clock::now();
    AssignPoints(n, data, k, assigned_clusters, distances);
    auto assign_end = std::chrono::high_resolution_clock::now();
    bm.cluster_search_time = std::chrono::duration_cast<std::chrono::microseconds>(assign_end - assign_beg).count();
  }

 public:
  Compass(size_t n, size_t d, size_t da, size_t M, size_t efc, size_t nlist)
      : HybridIndex<dist_t, attr_t>(n, d, M, efc, nlist),
        btrees_(nlist, vector<btree::btree_map<attr_t, labeltype>>(da)),
        da_(da) {}

  void AddPointsToIvf(const size_t n, const dist_t *data, const labeltype *labels, const attr_t *attrs) override {
    AssignPoints(n, data, 1, this->base_cluster_rank_);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < da_; j++) {
        btrees_[this->base_cluster_rank_[i]][j].insert(std::make_pair(attrs[i * da_ + j], labels[i]));
      }
    }
  }

  void LoadRanking(fs::path path, attr_t *attrs) override {
    std::ifstream in(path.string());
    faiss::idx_t assigned_cluster;
    for (int i = 0; i < this->n_; i++) {
      in.read((char *)(&assigned_cluster), sizeof(faiss::idx_t));
      for (int j = 0; j < da_; j++) {
        btrees_[assigned_cluster][j].insert(std::make_pair(attrs[i * da_ + j], (labeltype)i));
      }
    }
  }

  // By default, we will not use the distances to centroids.
  vector<vector<pair<float, hnswlib::labeltype>>> SearchKnn(
      const std::variant<const dist_t *, pair<const dist_t *, const dist_t *>> &var,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) override {
    auto efs_ = std::max(k, efs);
    this->hnsw_.setEf(efs_);
    int nprobe = this->nlist_ / 20;

    const dist_t *query, *xquery;
    if (std::holds_alternative<const dist_t *>(var)) {
      query = std::get<const dist_t *>(var);
      xquery = query;
    } else {
      query = std::get<pair<const dist_t *, const dist_t *>>(var).first;
      xquery = std::get<pair<const dist_t *, const dist_t *>>(var).second;
    }
    SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<vector<pair<dist_t, labeltype>>> results(nq, vector<pair<dist_t, labeltype>>(k));
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;
      priority_queue<pair<float, int64_t>> recycle_set;

      vector<bool> visited(this->n_, false);

      bm.qmetrics[q].nround = 0;
      bm.qmetrics[q].ncomp = 0;

      int curr_ci = q * nprobe;

      std::vector<std::unordered_set<labeltype>> candidates_per_dim(da_);
      for (int j = 0; j < da_; ++j) {
        auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][j];
        auto beg = btree.lower_bound(l_bound[j]);
        auto end = btree.upper_bound(u_bound[j]);
        for (auto itr = beg; itr != end; ++itr) {
          candidates_per_dim[j].insert(itr->second);
        }
      }
      // Intersect all sets in candidates_per_dim
      std::unordered_set<labeltype> intersection;
      if (da_ > 0) intersection = candidates_per_dim[0];
      for (int j = 1; j < da_; ++j) {
        std::unordered_set<labeltype> temp;
        for (const auto &id : intersection) {
          if (candidates_per_dim[j].count(id)) {
            temp.insert(id);
          }
        }
        intersection = std::move(temp);
      }

      auto itr_beg = intersection.begin();
      auto itr_end = intersection.end();

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (curr_ci < nprobe * (q + 1))) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              curr_ci++;
              if (curr_ci >= (q + 1) * nprobe)
                break;
              else {
                std::vector<std::unordered_set<labeltype>> _candidates_per_dim(da_);
                for (int j = 0; j < da_; ++j) {
                  auto &btree = this->btrees_[this->query_cluster_rank_[curr_ci]][j];
                  auto beg = btree.lower_bound(l_bound[j]);
                  auto end = btree.upper_bound(u_bound[j]);
                  for (auto itr = beg; itr != end; ++itr) {
                    _candidates_per_dim[j].insert(itr->second);
                  }
                }
                // Intersect all sets in candidates_per_dim
                if (da_ > 0) intersection = _candidates_per_dim[0];
                for (int j = 1; j < da_; ++j) {
                  std::unordered_set<labeltype> temp;
                  for (const auto &id : intersection) {
                    if (_candidates_per_dim[j].count(id)) {
                      temp.insert(id);
                    }
                  }
                  intersection = std::move(temp);
                }
                itr_beg = intersection.begin();
                itr_end = intersection.end();
                continue;
              }
            }

            auto tableid = *itr_beg;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->hnsw_.getDataByInternalId(*itr_beg), _MM_HINT_T0);
#endif
            if (visited[tableid]) continue;

            auto vect = this->hnsw_.getDataByInternalId(tableid);
            auto dist = this->hnsw_.fstdistfunc_((float *)query + q * this->d_, vect, this->hnsw_.dist_func_param_);
            bm.qmetrics[q].ncomp++;
            crel++;

            recycle_set.emplace(-dist, tableid);
          }
          bm.qmetrics[q].nround++;
          int cnt = this->hnsw_.M_;
          while (!recycle_set.empty() && cnt > 0) {
            auto top = recycle_set.top();
            recycle_set.pop();
            if (visited[top.second]) continue;
            visited[top.second] = true;
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            candidate_set.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            if (top_candidates.size() > efs_) top_candidates.pop();  // better not to overflow the result queue
            cnt--;
          }
        }

        this->hnsw_.ReentrantSearchKnnBounded(
            (float *)query + q * this->d_,
            k,
            -recycle_set.top().first,  // cause infinite loop?
            // distances[curr_ci],
            top_candidates,
            candidate_set,
            visited,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || curr_ci >= (q + 1) * nprobe) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = curr_ci - q * nprobe;
      int nrecycled = 0;
      while (top_candidates.size() > k) top_candidates.pop();
      while (!recycle_set.empty()) {
        auto top = recycle_set.top();
        if (top_candidates.size() >= k && -top.first > top_candidates.top().first)
          break;
        else {
          top_candidates.emplace(-top.first, top.second);
          bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
          if (top_candidates.size() > k) top_candidates.pop();
          nrecycled++;
        }
        recycle_set.pop();
      }
      bm.qmetrics[q].nrecycled = nrecycled;
      while (top_candidates.size() > k) top_candidates.pop();
      size_t sz = top_candidates.size();
      while (!top_candidates.empty()) {
        results[q][--sz] = top_candidates.top();
        top_candidates.pop();
      }
    }

    return results;
  }
};