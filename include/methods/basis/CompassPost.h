#include <fmt/core.h>
#include <array>
#include <boost/filesystem.hpp>
#include "faiss/Index.h"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/index_io.h"
#include "fc/btree.h"
#include "hnswlib/hnswlib.h"
#include "methods/basis/IterativeSearch.h"
#include "roaring/roaring.hh"
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
  vector<fc::BTreeMultiMap<attr_t, pair<labeltype, array<attr_t, 4>>, 32>> btrees_;
  vector<fc::BTreeMultiMap<attr_t, labeltype>> mbtrees_;
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
        mbtrees_(nlist * da),
        base_cluster_rank_(new faiss::idx_t[n]) {
    cg_.SetSearchParam(batch_k, initial_efs, delta_efs);
    ivf_ = new faiss::IndexIVFFlat(new faiss::IndexFlatL2(d), d, nlist);
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
  virtual void AddPointsToIvf(const size_t n, const void *data, const labeltype *labels, attr_t *attrs) {
    AssignPoints(n, data, 1, this->base_cluster_rank_);
    for (int i = 0; i < n; i++) {
      array<attr_t, 4> arr{0, 0, 0, 0};
      for (int j = 0; j < this->da_; j++) {
        arr[j] = attrs[i * this->da_ + j];
        mbtrees_[this->base_cluster_rank_[i] * this->da_ + j].insert(
            frozenca::BTreePair<attr_t, labeltype>(std::move(attrs[i * this->da_ + j]), (labeltype)i)
        );
      }
      btrees_[this->base_cluster_rank_[i]].insert(frozenca::BTreePair<attr_t, pair<labeltype, array<attr_t, 4>>>(
          std::move(attrs[i * this->da_]), {(labeltype)i, arr}
      ));
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
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
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
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            // if (vl->mass[tableid] == vl->curV) {
            //   continue;
            // }
            // vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          for (int i = 0; i < k / 2 && !top_ivf.empty(); i++) {
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

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFiltered(
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

    for (int q = 0; q < nq; q++) {
#ifndef BENCH
      auto q_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      vl->reset();
      priority_queue<pair<dist_t, labeltype>> top_candidates;  // storing the final filtered result
      priority_queue<pair<dist_t, labeltype>> top_ivf;
      const void *query_q = (char *)query + (q * graph_.hnsw_->data_size_);
#ifndef BENCH
      auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      graph_.SetSearchParam(k, k, k);
      auto state = graph_.OpenFiltered(query_q, graph_.hnsw_->max_elements_, vl, &pred);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      while (top_candidates.size() < efs) {
        // TODO: 0.1 is a tuning point.
        // If this number is too small, we have to rely on graph to add points slowly
        // Otherwise, we will turn to IVF for batch addition (k in our case).
        if ((nround_graph >= 1 && num_graph_ppsl <= nround_graph * k * 0.2)) {
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
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
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
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          for (int i = 0; i < k && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            state.candidate_set_.emplace(top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
            top_candidates.push(std::make_pair(-top.first, top.second));
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            num_ivf_ppsl++;
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
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatchFiltered(&state, &pred);
          num_graph_ppsl += batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            // if (pred(label)) {
            top_candidates.push(std::make_pair(-dist, label));
            bm.qmetrics[q].is_graph_ppsl[label] = true;
            // }
          }
          nround_graph++;
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

  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHop(
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

    // graph_.SetSearchParam(20, 20, k);
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
      // Enlarge the search to reduce overhead.
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      // graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg, cg_.batch_k_);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
            initialized = true;
            clus_cnt++;
          }
          bool restart = !state.candidate_set_.empty() && !top_ivf.empty() &&
                         (state.result_set_.empty() || -top_ivf.top().first > -state.result_set_.top().first);
          if (restart) {
            state.sel_ = 1;        // restart graph
            graph_last_round = 1;  // restart graph
          }
          if (restart && !state.result_set_.empty()) {
            continue;  // restart directly
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
#ifndef BENCH
              auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
#ifndef BENCH
              auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
              auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
              bm.qmetrics[q].filter_latency += btree_time;
#endif
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            // Should prioritize the graph search? No idea yet... Leave it this first.
            // vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.emplace(-dist, tableid);
            crel++;
          }
          int i = 0;
          // Restart is good with graph early stopping.
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            if (vl->mass[top.second] == vl->curV) {
              continue;
            }
            vl->mass[top.second] = vl->curV;
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
#endif
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          graph_.NextBatchTwoHop(&state, &pred);
          priority_queue<pair<dist_t, labeltype>> &batch = state.result_set_;
          int i = 0;
          while (!batch.empty() && i < graph_.batch_k_) {
            auto top = batch.top();
            batch.pop();
            i++;
            top_candidates.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_graph_ppsl[top.second] = true;
#endif
          }
          num_graph_ppsl += i;
          graph_last_round = i;
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
#ifndef BENCH
      fmt::print("twohop_count: {}\n", state.out_.twohop_count);
      bm.qmetrics[q].nrecycled += state.out_.checked_count;
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      bm.qmetrics[q].comp_latency += state.out_.comp_time;
      bm.qmetrics[q].filter_latency += state.out_.filter_time;
#endif
      // graph_.Close(&state);
      // cg_.Close(&cg_state);
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

  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopRevision(
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

    VisitedList *vl = this->graph_.hnsw_->visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->cg_.hnsw_->visited_list_pool_->getFreeVisitedList();

    // graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
      RangeQuery<attr_t> pred(l_bound + q * this->da_, u_bound + q * this->da_, attrs, this->n_, this->da_);
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
      // Enlarge the search to reduce overhead.
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      // graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg, cg_.batch_k_);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
            initialized = true;
            clus_cnt++;
          }
          bool restart = !state.candidate_set_.empty() && !top_ivf.empty() &&
                         (state.result_set_.empty() || -top_ivf.top().first > -state.result_set_.top().first);
          if (restart) {
            state.sel_ = 1;        // restart graph
            graph_last_round = 1;  // restart graph
          }
          if (restart && !state.result_set_.empty()) {
            continue;  // restart directly
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
#ifndef BENCH
              auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
#ifndef BENCH
              auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
              auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
              bm.qmetrics[q].filter_latency += btree_time;
#endif
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            // Should prioritize the graph search? No idea yet... Leave it this first.
            // vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.emplace(-dist, tableid);
            crel++;
          }
          int i = 0;
          // Restart is good with graph early stopping.
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            if (vl->mass[top.second] == vl->curV) {
              continue;
            }
            vl->mass[top.second] = vl->curV;
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
#endif
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          graph_.NextBatchTwoHop(&state, &pred);
          priority_queue<pair<dist_t, labeltype>> &batch = state.result_set_;
          int i = 0;
          while (!batch.empty() && i < graph_.batch_k_) {
            auto top = batch.top();
            batch.pop();
            i++;
            top_candidates.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_graph_ppsl[top.second] = true;
#endif
          }
          num_graph_ppsl += i;
          graph_last_round = i;
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
#ifndef BENCH
      fmt::print("twohop_count: {}\n", state.out_.twohop_count);
      bm.qmetrics[q].nrecycled += state.out_.checked_count;
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      bm.qmetrics[q].comp_latency += state.out_.comp_time;
      bm.qmetrics[q].filter_latency += state.out_.filter_time;
#endif
      // graph_.Close(&state);
      // cg_.Close(&cg_state);
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

  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopCheating(
      const void *query,
      const int nq,
      const int k,
      const attr_t *attrs,  // The first dimension is a placeholder for vector id.
      const int32_t *l_ranges,
      const int32_t *u_ranges,
      const attr_t *l_bounds,  // one dimension less than attrs
      const attr_t *u_bounds,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);

    VisitedList *vl = this->graph_.hnsw_->visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->cg_.hnsw_->visited_list_pool_->getFreeVisitedList();

    // graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
      InplaceRangeQuery<attr_t> pred(l_ranges[q], u_ranges[q], l_bounds, u_bounds, attrs, this->n_, this->da_);
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
      // Enlarge the search to reduce overhead.
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      // graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg, cg_.batch_k_);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            itr_beg = btrees_[clus].lower_bound(this->da_ > 1 ? l_bounds[0] : l_ranges[q]);
            itr_end = btrees_[clus].upper_bound(this->da_ > 1 ? u_bounds[0] : (u_ranges[q] + 0.1));
            while (itr_beg != itr_end) {
              if (pred(itr_beg->second.first)) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
            initialized = true;
            clus_cnt++;
          }
          bool restart = !state.candidate_set_.empty() && !top_ivf.empty() &&
                         (state.result_set_.empty() || -top_ivf.top().first > -state.result_set_.top().first);
          if (restart) {
            state.sel_ = 1;        // restart graph
            graph_last_round = 1;  // restart graph
          }
          if (restart && !state.result_set_.empty()) {
            continue;  // restart directly
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
#ifndef BENCH
              auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              itr_beg = btrees_[clus].lower_bound(this->da_ > 1 ? l_bounds[0] : l_ranges[q]);
              itr_end = btrees_[clus].upper_bound(this->da_ > 1 ? u_bounds[0] : (u_ranges[q] + 0.1));
              while (itr_beg != itr_end) {
                if (pred(itr_beg->second.first)) {
                  break;
                } else {
                  itr_beg++;
                }
              }
#ifndef BENCH
              auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
              auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
              bm.qmetrics[q].filter_latency += btree_time;
#endif
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            while (itr_beg != itr_end) {
              if (pred(itr_beg->second.first)) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            // Should prioritize the graph search? No idea yet... Leave it this first.
            // vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.emplace(-dist, tableid);
            crel++;
          }
          int i = 0;
          // Restart is good with graph early stopping.
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            if (vl->mass[top.second] == vl->curV) {
              continue;
            }
            vl->mass[top.second] = vl->curV;
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
#endif
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          graph_.NextBatchTwoHop(&state, &pred);
          priority_queue<pair<dist_t, labeltype>> &batch = state.result_set_;
          int i = 0;
          while (!batch.empty() && i < graph_.batch_k_) {
            auto top = batch.top();
            batch.pop();
            i++;
            top_candidates.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_graph_ppsl[top.second] = true;
#endif
          }
          num_graph_ppsl += i;
          graph_last_round = i;
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
#ifndef BENCH
      fmt::print("twohop_count: {}\n", state.out_.twohop_count);
      bm.qmetrics[q].nrecycled += state.out_.checked_count;
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      bm.qmetrics[q].comp_latency += state.out_.comp_time;
      bm.qmetrics[q].filter_latency += state.out_.filter_time;
#endif
      // graph_.Close(&state);
      // cg_.Close(&cg_state);
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

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopGivenBitset(
      const void *query,
      const int nq,
      const int k,
      const attr_t *attrs,
      const attr_t *l_bound,
      const attr_t *u_bound,
      BitsetQuery<attr_t> *bitset,
      const int efs,
      const int nrel,
      const int nthread,
      BatchMetric &bm
  ) {
    vector<priority_queue<pair<dist_t, labeltype>>> results(nq);

    // RangeQuery<attr_t> pred(l_bound, u_bound, attrs, this->n_, this->da_);
    VisitedList *vl = this->graph_.hnsw_->visited_list_pool_->getFreeVisitedList();
    VisitedList *vl_cg = this->cg_.hnsw_->visited_list_pool_->getFreeVisitedList();

    // graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
      auto q_start = std::chrono::high_resolution_clock::system_clock::now();
      vl->reset();
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      priority_queue<pair<dist_t, labeltype>> top_ivf;
      const void *query_q = (char *)query + (q * graph_.hnsw_->data_size_);
      auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, bitset, vl);
      graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
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
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end) {
              if ((*bitset)(itr_beg->second.first)) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
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
#ifndef BENCH
              auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              while (itr_beg != itr_end) {
                if ((*bitset)(itr_beg->second.first)) {
                  break;
                } else {
                  itr_beg++;
                }
              }
#ifndef BENCH
              auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
              auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
              bm.qmetrics[q].filter_latency += btree_time;
#endif
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            while (itr_beg != itr_end) {
              if ((*bitset)(itr_beg->second.first)) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          int i = 0;
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            // state.result_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            vl->mass[top.second] = vl->curV;
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
          state.sel_ = 1;        // restart graph
          graph_last_round = 1;  // restart graph
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
          continue;
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatchTwoHop(&state, bitset);
          num_graph_ppsl += batch.size();
          graph_last_round = batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            // if (pred(label)) {
            top_candidates.push(std::make_pair(-dist, label));
            bm.qmetrics[q].is_graph_ppsl[label] = true;
            // }
          }
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
#ifndef BENCH
      bm.qmetrics[q].nrecycled += state.out_.checked_count;
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      bm.qmetrics[q].comp_latency += state.out_.comp_time;
      bm.qmetrics[q].filter_latency += state.out_.filter_time;
#endif
      // graph_.Close(&state);
      // cg_.Close(&cg_state);
      while (top_candidates.size() > k) top_candidates.pop();
      results[q] = std::move(top_candidates);
      auto q_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto q_time = std::chrono::duration_cast<std::chrono::nanoseconds>(q_stop - q_start).count();
      bm.qmetrics[q].latency = q_time;
    }
    return results;
  }

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopBit(
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
    roaring::Roaring bitmaps[4];
    vector<uint32_t> ans;

    // graph_.SetSearchParam(20, 20, k);
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
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(ans.begin()) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
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
            for (int i = 0; i < this->da_; i++) {
              bitmaps[i].removeRange(0, this->n_);
            }
            int clus = next.second;
            auto beg = mbtrees_[clus * this->da_ + 0].lower_bound(l_bound[0]);
            auto end = mbtrees_[clus * this->da_ + 0].upper_bound(u_bound[0]);
            while (beg != end) {
              bitmaps[0].add(beg->second);
              beg++;
            }
            for (int i = 1; i < this->da_; i++) {
              auto beg = mbtrees_[clus * this->da_ + i].lower_bound(l_bound[i]);
              auto end = mbtrees_[clus * this->da_ + i].upper_bound(u_bound[i]);
              while (beg != end) {
                bitmaps[i].add(beg->second);
                beg++;
              }
              bitmaps[0] &= bitmaps[i];
            }
            ans.reserve(bitmaps[0].cardinality());
            bitmaps[0].toUint32Array(ans.data());
            itr_beg = ans.begin();
            itr_end = ans.end();
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
              for (int i = 0; i < this->da_; i++) {
                bitmaps[i].removeRange(0, this->n_);
              }
              auto beg = mbtrees_[clus * this->da_ + 0].lower_bound(l_bound[0]);
              auto end = mbtrees_[clus * this->da_ + 0].upper_bound(u_bound[0]);
              while (beg != end) {
                bitmaps[0].add(beg->second);
                beg++;
              }
              for (int i = 1; i < this->da_; i++) {
                auto beg = mbtrees_[clus * this->da_ + i].lower_bound(l_bound[i]);
                auto end = mbtrees_[clus * this->da_ + i].upper_bound(u_bound[i]);
                while (beg != end) {
                  bitmaps[i].add(beg->second);
                  beg++;
                }
                bitmaps[0] &= bitmaps[i];
              }
              ans.reserve(bitmaps[0].cardinality());
              bitmaps[0].toUint32Array(ans.data());
              itr_beg = ans.begin();
              itr_end = ans.end();
              clus_cnt++;
              continue;
            }
            tableint tableid = *itr_beg;
            itr_beg++;
#ifdef USE_SSE
            if (itr_beg != itr_end) _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(*itr_beg), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          int i = 0;
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            // state.result_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            vl->mass[top.second] = vl->curV;
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
          state.sel_ = 1;  // restart graph
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
          continue;
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatchTwoHop(&state, &pred);
          num_graph_ppsl += batch.size();
          graph_last_round = batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            // if (pred(label)) {
            top_candidates.push(std::make_pair(-dist, label));
            bm.qmetrics[q].is_graph_ppsl[label] = true;
            // }
          }
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);

      graph_.Close(&state);
      cg_.Close(&cg_state);
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

  virtual vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredNavix(
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

    // graph_.SetSearchParam(20, 20, k);
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
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenNavix(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
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
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
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
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.push(std::make_pair(-dist, tableid));
            crel++;
          }
          int i = 0;
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            // state.result_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
            vl->mass[top.second] = vl->curV;
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
          state.sel_ = 1;  // restart graph
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
          continue;
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          priority_queue<pair<dist_t, labeltype>> batch = graph_.NextBatchNavix(&state, &pred);
          num_graph_ppsl += batch.size();
          graph_last_round = batch.size();
          while (!batch.empty()) {
            auto [dist, label] = batch.top();
            batch.pop();
            // if (pred(label)) {
            top_candidates.push(std::make_pair(-dist, label));
            bm.qmetrics[q].is_graph_ppsl[label] = true;
            // }
          }
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);

      graph_.Close(&state);
      cg_.Close(&cg_state);
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

  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopAlbationRelational(
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

    // graph_.SetSearchParam(20, 20, k);
    // cg_.SetSearchParam(20, 20, 20);

    for (int q = 0; q < nq; q++) {
#ifndef BENCH
      auto q_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
      vl->reset();
      priority_queue<pair<dist_t, labeltype>> top_candidates;
      const void *query_q = (char *)query + (q * graph_.hnsw_->data_size_);

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;
      int nround = 0;

      while (top_candidates.size() < efs) {
#ifndef BENCH
        auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
        if (!initialized) {
          vl_cg->reset();
#ifndef BENCH
          auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg, cg_.batch_k_);
          auto next = cg_.Next(&cg_state);
#ifndef BENCH
          auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
          bm.qmetrics[q].cg_latency += cg_time;
#endif
          int clus = next.second;
#ifndef BENCH
          auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          itr_beg = btrees_[clus].lower_bound(l_bound[0]);
          itr_end = btrees_[clus].upper_bound(u_bound[0]);
          while (itr_beg != itr_end) {
            auto arr = itr_beg->second.second;
            bool good = true;
            for (int i = 1; i < this->da_; i++) {
              if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                good = false;
                break;
              }
            }
            if (good) {
              break;
            } else {
              itr_beg++;
            }
          }
#ifndef BENCH
          auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
          auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
          bm.qmetrics[q].filter_latency += btree_time;
#endif
          initialized = true;
          clus_cnt++;
        }

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
#ifndef BENCH
          auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          itr_beg = btrees_[clus].lower_bound(l_bound[0]);
          itr_end = btrees_[clus].upper_bound(u_bound[0]);
          while (itr_beg != itr_end) {
            auto arr = itr_beg->second.second;
            bool good = true;
            for (int i = 1; i < this->da_; i++) {
              if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                good = false;
                break;
              }
            }
            if (good) {
              break;
            } else {
              itr_beg++;
            }
          }
#ifndef BENCH
          auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
          auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
          bm.qmetrics[q].filter_latency += btree_time;
#endif
          clus_cnt++;
          continue;
        }
        tableint tableid = itr_beg->second.first;
        itr_beg++;
#ifndef BENCH
        auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
        while (itr_beg != itr_end) {
          auto arr = itr_beg->second.second;
          bool good = true;
          for (int i = 1; i < this->da_; i++) {
            if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
              good = false;
              break;
            }
          }
          if (good) {
            break;
          } else {
            itr_beg++;
          }
        }
#ifndef BENCH
        auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
        auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
        bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
        if (itr_beg != itr_end)
          _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
        if (vl->mass[tableid] == vl->curV) {
          continue;
        }
        // Should prioritize the graph search? No idea yet... Leave it this first.
        // vl->mass[tableid] = vl->curV;
        auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
        auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
        bm.qmetrics[q].ncomp++;
        top_candidates.emplace(dist, tableid);

#ifndef BENCH
        auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
        auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
        bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        nround++;
      }

      bm.qmetrics[q].ncomp += 0;  // No longer add from graph.
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
      // #ifndef BENCH
      //       fmt::print("twohop_count: {}\n", state.out_.twohop_count);
      //       bm.qmetrics[q].nrecycled += state.out_.checked_count;
      //       bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      //       bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      //       bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      //       bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      //       bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      //       bm.qmetrics[q].comp_latency += state.out_.comp_time;
      //       bm.qmetrics[q].filter_latency += state.out_.filter_time;
      // #endif

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

  vector<priority_queue<pair<dist_t, labeltype>>> SearchKnnPostFilteredTwoHopAblationGraph(
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

    // graph_.SetSearchParam(20, 20, k);
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
      // Enlarge the search to reduce overhead.
      graph_.SetSearchParam(k, k, k);
      // graph_.SetSearchParam(k, efs, k); // For testing non-iterative version.
      auto state = graph_.OpenTwoHop(query_q, graph_.hnsw_->max_elements_, &pred, vl);
      // graph_.SetSearchParam(k / 2, k + k / 2, k / 2);
#ifndef BENCH
      auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
      auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
      bm.qmetrics[q].graph_latency += graph_time;
#endif

      decltype(btrees_[0].lower_bound(0)) itr_beg, itr_end;
      IterativeSearchState<dist_t> cg_state(query_q, k);
      bool initialized = false;
      int clus_cnt = 0;

      int nround_graph = 0, num_graph_ppsl = 0, nround = 0;
      int num_ivf_ppsl = 0;
      int graph_last_round = 0;
      double breaktie = 0.05;
      while (top_candidates.size() < efs) {
        // while (top_candidates.size() < k) { // For testing non-iterative version.
        // IVF is responsible for negative clustering and extremely low passrate.
        // Otherwise, post-filtering on graph should do.
        if ((nround_graph >= 1 && (state.sel_ <= breaktie || graph_last_round == 0))) {
#ifndef BENCH
          auto ivf_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          if (!initialized) {
            vl_cg->reset();
#ifndef BENCH
            auto cg_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            cg_state = cg_.Open(query_q, cg_.hnsw_->max_elements_, vl_cg, cg_.batch_k_);
            auto next = cg_.Next(&cg_state);
#ifndef BENCH
            auto cg_stop = std::chrono::high_resolution_clock::system_clock::now();
            auto cg_time = std::chrono::duration_cast<std::chrono::nanoseconds>(cg_stop - cg_start).count();
            bm.qmetrics[q].cg_latency += cg_time;
#endif
            int clus = next.second;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            itr_beg = btrees_[clus].lower_bound(l_bound[0]);
            itr_end = btrees_[clus].upper_bound(u_bound[0]);
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
            initialized = true;
            clus_cnt++;
          }
          bool restart = !state.candidate_set_.empty() && !top_ivf.empty() &&
                         (state.result_set_.empty() || -top_ivf.top().first > -state.result_set_.top().first);
          if (restart) {
            state.sel_ = 1;        // restart graph
            graph_last_round = 1;  // restart graph
          }
          if (restart && !state.result_set_.empty()) {
            continue;  // restart directly
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
#ifndef BENCH
              auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
              itr_beg = btrees_[clus].lower_bound(l_bound[0]);
              itr_end = btrees_[clus].upper_bound(u_bound[0]);
              while (itr_beg != itr_end) {
                auto arr = itr_beg->second.second;
                bool good = true;
                for (int i = 1; i < this->da_; i++) {
                  if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                    good = false;
                    break;
                  }
                }
                if (good) {
                  break;
                } else {
                  itr_beg++;
                }
              }
#ifndef BENCH
              auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
              auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
              bm.qmetrics[q].filter_latency += btree_time;
#endif
              clus_cnt++;
              continue;
            }
            tableint tableid = itr_beg->second.first;
            itr_beg++;
#ifndef BENCH
            auto btree_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
            while (itr_beg != itr_end) {
              auto arr = itr_beg->second.second;
              bool good = true;
              for (int i = 1; i < this->da_; i++) {
                if (arr[i] < l_bound[i] || arr[i] > u_bound[i]) {
                  good = false;
                  break;
                }
              }
              if (good) {
                break;
              } else {
                itr_beg++;
              }
            }
#ifndef BENCH
            auto btree_end = std::chrono::high_resolution_clock::system_clock::now();
            auto btree_time = std::chrono::duration_cast<std::chrono::nanoseconds>(btree_end - btree_start).count();
            bm.qmetrics[q].filter_latency += btree_time;
#endif
#ifdef USE_SSE
            if (itr_beg != itr_end)
              _mm_prefetch(this->graph_.hnsw_->getDataByInternalId(itr_beg->second.first), _MM_HINT_T0);
#endif
            if (vl->mass[tableid] == vl->curV) {
              continue;
            }
            // Should prioritize the graph search? No idea yet... Leave it this first.
            // vl->mass[tableid] = vl->curV;
            auto vect = this->graph_.hnsw_->getDataByInternalId(tableid);
            auto dist = this->graph_.hnsw_->fstdistfunc_(query_q, vect, this->graph_.hnsw_->dist_func_param_);
            bm.qmetrics[q].ncomp++;
            top_ivf.emplace(-dist, tableid);
            crel++;
          }
          int i = 0;
          // Restart is good with graph early stopping.
          for (; i < k / 2 && !top_ivf.empty(); i++) {
            auto top = top_ivf.top();
            top_ivf.pop();
            if (vl->mass[top.second] == vl->curV) {
              continue;
            }
            vl->mass[top.second] = vl->curV;
            // TODO: consider bounding by the top of top_candidates
            state.candidate_set_.emplace(top.first, top.second);
            top_candidates.emplace(-top.first, top.second);
            state.top_candidates_.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_ivf_ppsl[top.second] = true;
#endif
            num_ivf_ppsl++;
          }
          graph_.hnsw_->setEf(graph_.hnsw_->ef_ + i);
#ifndef BENCH
          auto ivf_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto ivf_time = std::chrono::duration_cast<std::chrono::nanoseconds>(ivf_stop - ivf_start).count();
          bm.qmetrics[q].ivf_latency += ivf_time;
#endif
        }
        // Believe in graph when the first-hop selectivity is not low.
        // "state.sel_ >=" means we do not always rely on graph.
        if (nround_graph == 0 || state.sel_ >= breaktie) {
#ifndef BENCH
          auto graph_start = std::chrono::high_resolution_clock::system_clock::now();
#endif
          graph_.NextBatchTwoHop(&state, &pred);
          priority_queue<pair<dist_t, labeltype>> &batch = state.result_set_;
          int i = 0;
          while (!batch.empty() && i < graph_.batch_k_) {
            auto top = batch.top();
            batch.pop();
            i++;
            top_candidates.emplace(-top.first, top.second);
#ifndef BENCH
            bm.qmetrics[q].is_graph_ppsl[top.second] = true;
#endif
          }
          num_graph_ppsl += i;
          graph_last_round = i;
#ifndef BENCH
          auto graph_stop = std::chrono::high_resolution_clock::system_clock::now();
          auto graph_time = std::chrono::duration_cast<std::chrono::nanoseconds>(graph_stop - graph_start).count();
          bm.qmetrics[q].graph_latency += graph_time;
#endif
          nround_graph++;
        }
        nround++;
      }

      bm.qmetrics[q].ncomp += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].ncomp_cg += this->cg_.GetNcomp(&cg_state);
      bm.qmetrics[q].nround = nround;
      bm.qmetrics[q].ncluster = clus_cnt;
#ifndef BENCH
      fmt::print("twohop_count: {}\n", state.out_.twohop_count);
      bm.qmetrics[q].nrecycled += state.out_.checked_count;
      bm.qmetrics[q].ncomp_graph += this->graph_.GetNcomp(&state);
      bm.qmetrics[q].twohop_latency += state.out_.twohop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.pop_time;
      bm.qmetrics[q].ihnsw_latency += state.out_.bk_time;
      bm.qmetrics[q].ihnsw_latency += cg_state.out_.pop_time;
      bm.qmetrics[q].comp_latency += state.out_.comp_time;
      bm.qmetrics[q].filter_latency += state.out_.filter_time;
#endif
      // graph_.Close(&state);
      // cg_.Close(&cg_state);
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
