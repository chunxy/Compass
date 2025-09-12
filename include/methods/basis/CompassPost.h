#include <array>
#include <boost/filesystem.hpp>
#include "faiss/Index.h"
#include "faiss/index_io.h"
#include "fc/btree.h"
#include "hnswlib/hnswlib.h"
#include "methods/basis/IterativeSearch.h"
#include "utils/Pod.h"

using std::array;
using std::pair;
using std::vector;

namespace fs = boost::filesystem;

template <typename dist_t, typename attr_t>
class CompassPost {
 protected:
  IterativeSearch<dist_t> graph_;
  IterativeSearch<dist_t> cg_;
  faiss::Index *ivf_;
  // vector<btree::btree_map<attr_t, labeltype>> btrees_;
  vector<fc::BTreeMap<dist_t, pair<labeltype, array<attr_t, 4>>, 32>> btrees_;
  faiss::idx_t *base_cluster_rank_;
  // faiss::idx_t *query_cluster_rank_;
  // dist_t *distances_;
  int n_, d_, M_, efc_, nlist_;
  int da_;

 public:
  CompassPost(
      size_t n,
      size_t d,
      size_t da,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t initial_efs,
      size_t delta_efs
  )
      : n_(n),
        d_(d),
        da_(da),
        M_(M),
        efc_(efc),
        nlist_(nlist),
        graph_(n, d, new L2Space(d), M),
        cg_(nlist, d, new L2Space(d), M_cg),
        btrees_(nlist),
        base_cluster_rank_(new faiss::idx_t[n]) {
    cg_.SetSearchParam(batch_k, initial_efs, delta_efs);
    ivf_ = nullptr;
  }

  virtual void SaveGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    graph_.hnsw_->saveIndex(path.string());
  }
  virtual void LoadGraph(fs::path path) { graph_.hnsw_->loadIndex(path.string(), new L2Space(d_)); }

  virtual void SaveIvf(fs::path path) {
    fs::create_directories(path.parent_path());
    faiss::write_index(dynamic_cast<faiss::Index *>(ivf_), path.c_str());
  }
  virtual void LoadIvf(fs::path path) {
    auto ivf_file = fopen(path.c_str(), "r");
    auto index = faiss::read_index(ivf_file);
    if (ivf_) delete ivf_;
    ivf_ = dynamic_cast<faiss::Index *>(index);
  }

  virtual void SaveRanking(fs::path path) {
    std::ofstream out(path.string());
    for (int i = 0; i < this->n_; i++) {
      out.write((char *)(this->base_cluster_rank_ + i), sizeof(faiss::idx_t));
    }
  }
  virtual void LoadRanking(fs::path path, attr_t *attrs) = 0;

  virtual void BuildClusterGraph() = 0;
  void SaveClusterGraph(fs::path path) {
    fs::create_directories(path.parent_path());
    this->cg_.hnsw_->saveIndex(path.string());
  }
  virtual void LoadClusterGraph(fs::path path) { this->cg_.hnsw_->loadIndex(path.string(), new L2Space(this->d_)); }

  virtual void TrainIvf(size_t n, const void *data) { ivf_->train(n, (float *)data); }
  virtual void AddPointsToGraph(const size_t n, const void *data, const labeltype *labels) {
    for (int i = 0; i < n; i++) {
      this->graph_.hnsw_->addPoint((char *)data + i * this->graph_.hnsw_->data_size_, labels[i], -1);
    }
  }
  virtual void AssignPoints(
      const size_t n,
      const void *data,
      const int k,
      faiss::idx_t *assigned_clusters,
      float *distances = nullptr
  ) = 0;
  virtual void AddPointsToIvf(const size_t n, const void *data, const labeltype *labels, const attr_t *attrs) {
    AssignPoints(n, data, 1, this->base_cluster_rank_);
    for (int i = 0; i < n; i++) {
      array<attr_t, 4> arr{0, 0, 0, 0};
      for (int j = 0; j < this->da_; j++) {
        arr[j] = attrs[i * this->da_ + j];
      }
      btrees_[this->base_cluster_rank_[i]][attrs[i]] = std::make_pair(labels[i], arr);
    }
  }

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnn(
      const void *query,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);

    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->graph_.hnsw_->visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->cg_.hnsw_->visited_list_pool_->getFreeVisitedList();

    graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
#ifndef BENCH
      auto q_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      vl->reset();
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> top_ivf;
      const void *query_q = (char *)query + (q * graph_.hnsw_->data_size_);
#ifndef BENCH
      auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      auto state = graph_.Open(query_q, graph_.hnsw_->max_elements_, vl);
      graph_.SetSearchParam(k, k, k);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int in_range_cnt = 0, total = 0, nround = 0;
      while (top_candidates.size() < efs) {
        if (total > 20 && in_range_cnt <= total * 0.1) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end && !pred(itr_beg->second.first)) {
              itr_beg++;
            }
            initialized = true;
            clus_cnt++;
          }

          int crel = 0;
          while (crel < nrel) {
            if (itr_beg == itr_end) {
#ifndef BENCH
              auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              auto next = cg_.Next(&cg_state);
#ifndef BENCH
              auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
              auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
              bm.qmetrics[q].cg_latency += cg_time;
#endif
              int clus = next.second;
              if (clus == -1) {
                break;
              }
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            do {
              itr_beg++;
            } while (itr_beg != itr_end && !pred(itr_beg->second.first));
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          for (int i = 0; i < k && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            top_candidates.push(std::make_pair(-top.first, top.second));
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            vl->mass[top.second] = vl->curV;
          }
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        } else {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatch(&state);
          total += batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            if (pred(label)) {
              top_candidates.push(std::make_pair(-dist, label));
              bm.qmetrics[q].is_graph_ppsl[label] = true;
              in_range_cnt++;
            }
          }
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
        }
        nround++;
      }

      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);

      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);
#ifndef BENCH
      auto q_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto q_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_stop - q_start).count();
      bm.qmetrics[q].latency = q_time;
#endif
    }
    return results;
  }

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnTwoHop(
      const void *query,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);

    RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->graph_.hnsw_->visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->cg_.hnsw_->visited_list_pool_->getFreeVisitedList();

    graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
#ifndef BENCH
      auto q_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      vl->reset();
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> top_ivf;
      const void *query_q = (char *)query + (q * graph_.hnsw_->data_size_);
#ifndef BENCH
      auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      auto state = graph_.Open(query_q, graph_.hnsw_->max_elements_, vl);
      graph_.SetSearchParam(k, k, k);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int in_range_cnt = 0, total = 0, nround = 0;
      while (top_candidates.size() < efs) {
        if (total > 20 && in_range_cnt <= total * 0.1) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end && !pred(itr_beg->second.first)) {
              itr_beg++;
            }
            initialized = true;
            clus_cnt++;
          }

          int crel = 0;
          while (crel < nrel) {
            if (itr_beg == itr_end) {
#ifndef BENCH
              auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              auto next = cg_.Next(&cg_state);
#ifndef BENCH
              auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
              auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
              bm.qmetrics[q].cg_latency += cg_time;
#endif
              int clus = next.second;
              if (clus == -1) {
                break;
              }
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            do {
              itr_beg++;
            } while (itr_beg != itr_end && !pred(itr_beg->second.first));
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          for (int i = 0; i < k && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            top_candidates.push(std::make_pair(-top.first, top.second));
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            vl->mass[top.second] = vl->curV;
          }
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        } else {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatch(&state);
          total += batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            if (pred(label)) {
              top_candidates.push(std::make_pair(-dist, label));
              bm.qmetrics[q].is_graph_ppsl[label] = true;
              in_range_cnt++;
            }
          }
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
        }
        nround++;
      }

      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);

      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);
#ifndef BENCH
      auto q_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto q_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_stop - q_start).count();
      bm.qmetrics[q].latency = q_time;
#endif
    }
    return results;
  }
};
