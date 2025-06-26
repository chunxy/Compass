#include <cstddef>
#include "Compass1d.h"
#include "IterativeSearch.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass1dIcg : public Compass1d<dist_t, attr_t> {
 protected:
  IterativeSearch<dist_t> *isearch_;

  void SearchClusters(
      const size_t n,
      const dist_t *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      BatchMetric &bm,
      float *distances = nullptr
  ) override {}  // dummy implementation

 public:
  // This index only loads the ReentrantHnsw but does not build it.
  Compass1dIcg(
      size_t n,
      size_t d,
      size_t M,
      size_t efc,
      size_t nlist,
      const string &path,
      size_t batch_k,
      size_t delta_efs
  )
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist) {
    this->isearch_ = new IterativeSearch<dist_t>(n, d, path, batch_k, delta_efs);
  }

  // By default, we will not use the distances to centroids.
  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnn(
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
    // SearchClusters(nq, xquery, nprobe, this->query_cluster_rank_, bm);

    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);
    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, 1);
    VisitedList *vl = this->hnsw_.visited_list_pool_->getFreeVisitedList();

    // #pragma omp parallel for num_threads(nthread) schedule(static)
    for (int q = 0; q < nq; q++) {
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> candidate_set;
      priority_queue<pair<dist_t, labeltype>> recycle_set;

      vl->reset();
      vl_type *visited = vl->mass;
      vl_type visited_tag = vl->curV;
      // vector<bool> visited(this->n_, false);

      auto state = isearch_->Open((float *)query + q * this->d_, nprobe);

      auto next = isearch_->Next(state);
      int clus = next.second, clus_cnt = 1;
      auto itr_beg = this->btrees_[clus].lower_bound(*l_bound);
      auto itr_end = this->btrees_[clus].upper_bound(*u_bound);

      while (true) {
        int crel = 0;
        if (candidate_set.empty() || (clus != -1)) {
          while (crel < nrel) {
            if (itr_beg == itr_end) {
              auto next = isearch_->Next(state);
              clus = next.second;
              clus_cnt++;
              if (clus == -1)
                break;
              else {
                itr_beg = this->btrees_[clus].lower_bound(*l_bound);
                itr_end = this->btrees_[clus].upper_bound(*u_bound);
                // recycle_set = priority_queue<pair<float, int64_t>>();
                continue;
              }
            }

            auto tableid = (*itr_beg).second;
            itr_beg++;
#ifdef USE_SSE
            _mm_prefetch(this->hnsw_.getDataByInternalId((*itr_beg).second), _MM_HINT_T0);
#endif
            if (visited[tableid] == visited_tag) continue;

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
            if (visited[top.second] == visited_tag) continue;
            visited[top.second] = visited_tag;
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
            vl,
            &pred,
            std::ref(bm.qmetrics[q].ncomp),
            std::ref(bm.qmetrics[q].is_graph_ppsl)
        );
        if ((top_candidates.size() >= efs_) || clus == -1) {
          break;
        }
      }

      bm.qmetrics[q].ncluster = clus_cnt;
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
      results[q] = std::move(top_candidates);

      bm.cluster_search_ncomp += isearch_->GetNcomp(state);
      isearch_->Close(state);
    }

    return results;
  }
};