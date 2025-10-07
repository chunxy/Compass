#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <string>
#include <vector>
#include "config.h"
#include "json.hpp"
#include "methods/CompassPostK.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;
using namespace std::chrono;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  IvfGraph2dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  assert(nq % args.batchsz == 0);

  std::string method = "CompassPostKThb";
  std::string workload = fmt::format(
      HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, fmt::join(args.l_bounds, "-"), fmt::join(args.u_bounds, "-"), args.k
  );
  std::string build_param = fmt::format("M_{}_efc_{}_nlist_{}_M_cg_{}", args.M, args.efc, args.nlist, args.M_cg);

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt(c, args.l_bounds, args.u_bounds, args.k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");

  // Compute selectivity.
  int nsat;
  stat_selectivity(attrs, nb, c.attr_dim, args.l_bounds, args.u_bounds, nsat);

  CompassPostK<float, float> comp(
      nb, d, c.attr_dim, args.M, args.efc, args.nlist, args.M_cg, args.batch_k, args.initial_efs, args.delta_efs
  );
  fs::path ckp_root(CKPS);
  std::string graph_ckp = fmt::format(COMPASS_GRAPH_CHECKPOINT_TMPL, args.M, args.efc);
  std::string ivf_ckp = fmt::format(COMPASS_IVF_CHECKPOINT_TMPL, args.nlist);
  std::string rank_ckp = fmt::format(COMPASS_RANK_CHECKPOINT_TMPL, nb, args.nlist);
  std::string cgraph_ckp = fmt::format(COMPASS_CGRAPH_CHECKPOINT_TMPL, args.nlist, args.M_cg, 200);
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

  std::vector<labeltype> labels(nb);
  std::iota(labels.begin(), labels.end(), 0);
  if (fs::exists(ckp_dir / rank_ckp)) {
    comp.LoadRanking(ckp_dir / rank_ckp, attrs);
    fmt::print("Finished loading IVF ranking.\n");
  } else {
    auto add_points_start = high_resolution_clock::now();
    comp.AddPointsToIvf(nb, xb, labels.data(), attrs);
    auto add_points_stop = high_resolution_clock::now();
    fmt::print(
        "Finished adding points, took {} microseconds.\n",
        duration_cast<microseconds>(add_points_stop - add_points_start).count()
    );
    comp.SaveRanking(ckp_dir / rank_ckp);
  }

  if (fs::exists(ckp_dir / graph_ckp)) {
    comp.LoadGraph((ckp_dir / graph_ckp).string());
    fmt::print("Finished loading graph index.\n");
  } else {
    auto build_index_start = high_resolution_clock::now();
    comp.AddPointsToGraph(nb, xb, labels.data());
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building graph, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    comp.SaveGraph(ckp_dir / graph_ckp);
  }

  if (fs::exists(ckp_dir / cgraph_ckp)) {
    comp.LoadClusterGraph((ckp_dir / cgraph_ckp).string());
    fmt::print("Finished loading cluster graph index.\n");
  } else {
    auto build_index_start = high_resolution_clock::now();
    comp.BuildClusterGraph();
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building cluster graph, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    comp.SaveClusterGraph(ckp_dir / cgraph_ckp);
  }
  fmt::print("Finished loading indices.\n");

  for (auto efs : args.efs) {
    for (auto nrel : args.nrel) {
      time_t ts = time(nullptr);
      auto tm = localtime(&ts);
      std::string search_param = fmt::format(
          "efs_{}_nrel_{}_batch_k_{}_initial_efs_{}_delta_efs_{}",
          efs,
          nrel,
          args.batch_k,
          args.initial_efs,
          args.delta_efs
      );
      std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
      std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
      // fs::path log_root(fmt::format(LOGS, args.k) + "_special");
      fs::path log_root(fmt::format(LOGS, args.k));
      fs::path log_dir = log_root / method / workload / build_param / search_param;
      fs::create_directories(log_dir);
      fmt::print("Saving to {}.\n", (log_dir / out_json).string());
      FILE *out = stdout;
      nq = args.fast ? 200 : nq;
#ifndef COMPASS_DEBUG
      fmt::print("Writing to {}.\n", (log_dir / out_text).string());
      out = fopen((log_dir / out_text).c_str(), "w");
#endif

      int nbatch = nq / args.batchsz;
      vector<BatchMetric> bms(nbatch, BatchMetric(args.batchsz, nb));
      vector<vector<priority_queue<pair<float, hnswlib::labeltype>>>> results(nbatch);
      long long search_time = 0;
#ifndef COMPASS_DEBUG
// #pragma omp parallel
// #pragma omp single
// #pragma omp taskloop
#endif
      for (int j = 0; j < nq; j += args.batchsz) {
        int b = j / args.batchsz;
        auto batch_start = high_resolution_clock::system_clock::now();
        results[b] = comp.SearchKnnPostFilteredTwoHopBit(
            xq + j * d,
            args.batchsz,
            args.k,
            attrs,
            args.l_bounds.data(),
            args.u_bounds.data(),
            efs,
            nrel,
            args.nthread,
            bms[b]
        );
        auto batch_stop = high_resolution_clock::system_clock::now();
        auto batch_time = duration_cast<microseconds>(batch_stop - batch_start).count();
        bms[b].time = batch_time;
        search_time += batch_time;
      }
      // statistics
      Stat stat(nq);
      for (int j = 0; j < nq; j += args.batchsz) {
        int b = j / args.batchsz;
        vector<float> gt_min_s(results[b].size()), gt_max_s(results[b].size());
        for (int i = 0; i < results[b].size(); i++) {
          gt_min_s[i] = dist_func(xq + (j + i) * d, xb + hybrid_topks[j + i].front() * d, &d);
          gt_max_s[i] = dist_func(xq + (j + i) * d, xb + hybrid_topks[j + i].back() * d, &d);
        }
        collect_batch_metric(results[b], bms[b], hybrid_topks, j, gt_min_s, gt_max_s, EPSILON, nsat, stat);
      }

      auto json = collate_stat(stat, nb, nsat, args.k, nq, search_time, args.nthread, out);
      std::ofstream ofs((log_dir / out_json).c_str());
      ofs.write(json.dump(4).c_str(), json.dump(4).length());
    }
  }
}