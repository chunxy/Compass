#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "config.h"
#include "faiss/MetricType.h"
#include "json.hpp"
#include "methods/CompassR1d.h"
#include "methods/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using std::vector;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  IvfGraph1dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  assert(nq % args.batchsz == 0);

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string method = "CompassCg1d";
  std::string workload = fmt::format(HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, args.l_bound, args.u_bound, args.k);
  std::string build_param = fmt::format("M_{}_efc_{}_nlist_{}", args.M, args.efc, args.nlist);
  std::string search_param = fmt::format("efs_{}_nrel_{}_mincomp_{}", args.efs, args.nrel, args.mincomp);
  std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path log_root(fmt::format(LOGS, args.k));
  fs::path log_dir = log_root / method / workload / build_param / search_param;
  fs::create_directories(log_dir);

  fmt::print("Saving to {}.\n", (log_dir / out_json).string());
#ifndef COMPASS_DEBUG
  fmt::print("Writing to {}.\n", (log_dir / out_text).string());
  freopen((log_dir / out_text).c_str(), "w", stdout);
#endif

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  vector<float> attrs(_attrs.size());
  for (size_t i = 0; i < attrs.size(); i++) attrs[i] = _attrs[i][0];
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt(c, {args.l_bound}, vector<float>{args.u_bound}, args.k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");

  // Compute selectivity.
  int nsat;
  stat_selectivity(attrs, args.l_bound, args.u_bound, nsat);

  CompassR1d<float, float> comp(d, args.M, args.efc, nb, args.nlist, args.nrel);
  fs::path ckp_root(CKPS);
  std::string graph_ckp = fmt::format(COMPASS_GRAPH_CHECKPOINT_TMPL, args.M, args.efc);
  std::string ivf_ckp = fmt::format(COMPASS_IVF_CHECKPOINT_TMPL, args.nlist);
  std::string rank_ckp = fmt::format(COMPASS_RANK_CHECKPOINT_TMPL, nb, args.nlist);
  std::string cluster_graph_ckp = fmt::format(COMPASS_RANK_CHECKPOINT_TMPL, args.M, args.efc, args.nlist);
  fs::path ckp_dir = ckp_root / "CompassR1d" / c.name;
  if (fs::exists(ckp_dir / ivf_ckp)) {
    comp.LoadIvf(ckp_dir / ivf_ckp);
    fmt::print("Finished loading IVF index.\n");
  } else {
    auto train_ivf_start = high_resolution_clock::now();
    comp.TrainIvf(nb, xb);
    auto train_ivf_stop = high_resolution_clock::now();
    fmt::print(
        "Finished training IVF, took {} microseconds.\n",
        duration_cast<microseconds>(train_ivf_stop - train_ivf_start).count()
    );
    comp.SaveIvf(ckp_dir / ivf_ckp);
  }

  // if (fs::exists(ckp_dir / rank_ckp)) {
  //   comp.LoadRanking(ckp_dir / rank_ckp, attrs.data());
  //   fmt::print("Finished loading cluster ranking.\n");
  // } else {
  //   auto add_points_start = high_resolution_clock::now();
  //   std::vector<labeltype> labels;
  //   labels.resize(nb);
  //   std::iota(labels.begin(), labels.end(), 0);
  //   comp.AddIvfPoints(nb, xb, labels.data(), attrs.data());
  //   auto add_points_stop = high_resolution_clock::now();
  //   fmt::print(
  //       "Finished adding points, took {} microseconds.\n",
  //       duration_cast<microseconds>(add_points_stop - add_points_start).count()
  //   );
  //   comp.SaveRanking(ckp_dir / rank_ckp);
  // }

  if (fs::exists(ckp_dir / graph_ckp)) {
    comp.LoadGraph((ckp_dir / graph_ckp).string());
    fmt::print("Finished loading graph index.\n");
  } else {
    auto build_index_start = high_resolution_clock::now();
    for (int i = 0; i < nb; i++) comp.AddGraphPoint(xb + i * d, i);
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building graph, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    comp.SaveGraph(ckp_dir / graph_ckp);
  }


  if (fs::exists(ckp_dir / cluster_graph_ckp)) {
    comp.LoadClusterGraph((ckp_dir / cluster_graph_ckp).string());
    fmt::print("Finished loading cluster graph index.\n");
  } else {
    auto build_index_start = high_resolution_clock::now();
    comp.BuildClusterGraph();
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building cluster graph, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    comp.SaveClusterGraph(ckp_dir / cluster_graph_ckp);
  }

  fmt::print("Finished loading indices.\n");

  vector<Metric> metrics(args.batchsz, Metric(nb));
  faiss::idx_t *ranked_clusters = new faiss::idx_t[args.batchsz * args.nlist];
  float *distances = new float[args.batchsz * args.nlist];
  auto search_start = high_resolution_clock::system_clock::now();
#ifndef COMPASS_DEBUG
// #pragma omp parallel
// #pragma omp single
// #pragma omp taskloop
#endif
  for (int j = 0; j < nq; j += args.batchsz) {
    comp.SearchKnnV4(
        xq + j * d,
        args.batchsz,
        args.k,
        args.l_bound,
        args.u_bound,
        args.efs,
        args.mincomp,
        args.nthread,
        metrics
    );
  }
  auto search_stop = high_resolution_clock::system_clock::now();
  auto search_time = duration_cast<microseconds>(search_stop - search_start).count();

  // statistics
  Stat stat(nq);
  for (int j = 0; j < nq;) {
    vector<Metric> metrics(args.batchsz, Metric(nb));
    auto search_start = high_resolution_clock::now();
    auto results = comp.SearchKnnV4(
        xq + j * d,
        args.batchsz,
        args.k,
        args.l_bound,
        args.u_bound,
        args.efs,
        args.mincomp,
        args.nthread,
        metrics
    );
    auto search_stop = high_resolution_clock::now();

    for (int ii = 0; ii < results.size(); ii++) {
      auto rz = results[ii];
      auto metric = metrics[ii];
      std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;
      int ivf_ppsl_in_rz = 0, graph_ppsl_in_rz = 0;
      for (auto pair : rz) {
        auto i = pair.second;
        auto d = pair.first;
        assert(rz_indices.find(i) == rz_indices.end());
        rz_indices.insert(i);
        if (metric.is_ivf_ppsl[i])
          ivf_ppsl_in_rz++;
        else if (metric.is_graph_ppsl[i])
          graph_ppsl_in_rz++;
      }

      int ivf_ppsl_in_tp = 0, graph_ppsl_in_tp = 0;
      for (auto i : hybrid_topks[j]) {
        gt_indices.insert(i);
        if (metric.is_ivf_ppsl[i])
          ivf_ppsl_in_tp++;
        else if (metric.is_graph_ppsl[i])
          graph_ppsl_in_tp++;
      }
      auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
      auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);

      std::set_intersection(
          gt_indices.begin(),
          gt_indices.end(),
          rz_indices.begin(),
          rz_indices.end(),
          std::inserter(rz_gt_interse, rz_gt_interse.begin())
      );

      stat.rz_min_s[j] = rz.empty() ? std::numeric_limits<float>::max() : rz.front().first;
      stat.rz_max_s[j] = rz.empty() ? std::numeric_limits<float>::max() : rz.back().first;
      stat.ivf_ppsl_in_rz_s[j] = ivf_ppsl_in_rz;
      stat.graph_ppsl_in_rz_s[j] = graph_ppsl_in_rz;
      stat.gt_min_s[j] = gt_min;
      stat.gt_max_s[j] = gt_max;
      stat.ivf_ppsl_in_tp_s[j] = ivf_ppsl_in_tp;
      stat.graph_ppsl_in_tp_s[j] = graph_ppsl_in_tp;

      stat.tp_s[j] = rz_gt_interse.size();
      stat.rz_s[j] = rz_indices.size();
      stat.rec_at_ks[j] = (double)rz_gt_interse.size() / gt_indices.size();
      stat.pre_at_ks[j] = (double)rz_gt_interse.size() / rz_indices.size();

      stat.ivf_ppsl_nums[j] = std::accumulate(metric.is_ivf_ppsl.begin(), metric.is_ivf_ppsl.end(), 0);
      stat.graph_ppsl_nums[j] = std::accumulate(metric.is_graph_ppsl.begin(), metric.is_graph_ppsl.end(), 0);
      stat.ivf_ppsl_qlty[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_tp / stat.ivf_ppsl_nums[j] : 0;
      stat.ivf_ppsl_rate[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_rz / stat.ivf_ppsl_nums[j] : 0;
      stat.graph_ppsl_qlty[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_tp / stat.graph_ppsl_nums[j] : 0;
      stat.graph_ppsl_rate[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_rz / stat.graph_ppsl_nums[j] : 0;
      stat.perc_of_ivf_ppsl_in_tp[j] = rz_gt_interse.size() != 0 ? (double)ivf_ppsl_in_tp / rz_gt_interse.size() : 0;
      stat.perc_of_ivf_ppsl_in_rz[j] = (double)ivf_ppsl_in_rz / rz.size();
      stat.linear_scan_rate[j] = (double)stat.ivf_ppsl_nums[j] / nsat;
      stat.num_computations[j] = metric.ncomp;
      stat.num_rounds[j] = metric.nround;
      stat.num_clusters[j] = metric.ncluster;
      stat.latencies[j] = duration_cast<microseconds>(search_stop - search_start).count();
      j++;
    }
  }

  auto json = collate_stat(stat, nb, nsat, args.k, nq, search_time, args.nthread);
  std::ofstream ofs((log_dir / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}