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
#include "methods/Compass1dXK.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;
using namespace std::chrono;

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

  std::string method = "CompassIdec";
  std::string workload = fmt::format(HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, args.l_bound, args.u_bound, args.k);
  std::string build_param = fmt::format("M_{}_efc_{}_nlist_{}_dx_{}", args.M, args.efc, args.nlist, args.dx);

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

  int dx = args.dx;
  Compass1dXK<float, float> comp(nb, d, dx, args.M, args.efc, args.nlist);
  fs::path ckp_root(CKPS);
  std::string graph_ckp = fmt::format(COMPASS_GRAPH_CHECKPOINT_TMPL, args.M, args.efc);
  std::string ivf_ckp = fmt::format(COMPASS_X_IVF_CHECKPOINT_TMPL, args.nlist, dx);
  std::string rank_ckp = fmt::format(COMPASS_X_RANK_CHECKPOINT_TMPL, nb, args.nlist, dx);
  fs::path ckp_dir = ckp_root / "CompassR1d" / c.name;

  fs::path data_root(DATA);
  auto xbx = load_float32((data_root / "idec" / fmt::format("{}-100-{}.base.float32", c.name, dx)).string(), nb, dx);
  auto xqx = load_float32((data_root / "idec" / fmt::format("{}-100-{}.query.float32", c.name, dx)).string(), nq, dx);
  if (fs::exists(ckp_root / "IDEC" / c.name / ivf_ckp)) {
    comp.LoadIvf(ckp_root / "IDEC" / c.name / ivf_ckp);
    fmt::print("Finished loading IVF index.\n");
  } else {
    auto train_ivf_start = high_resolution_clock::now();
    comp.TrainIvf(nb, xbx);
    auto train_ivf_stop = high_resolution_clock::now();
    fmt::print(
        "Finished training IVF, took {} microseconds.\n",
        duration_cast<microseconds>(train_ivf_stop - train_ivf_start).count()
    );
    comp.SaveIvf(ckp_root / "IDEC" / c.name / ivf_ckp);
  }

  std::vector<labeltype> labels(nb);
  std::iota(labels.begin(), labels.end(), 0);
  if (fs::exists(ckp_root / "IDEC" / c.name / rank_ckp)) {
    comp.LoadRanking(ckp_root / "IDEC" / c.name / rank_ckp, attrs.data());
    fmt::print("Finished loading IVF ranking.\n");
  } else {
    auto add_points_start = high_resolution_clock::now();
    comp.AddPointsToIvf(nb, xbx, labels.data(), attrs.data());
    auto add_points_stop = high_resolution_clock::now();
    fmt::print(
        "Finished adding points, took {} microseconds.\n",
        duration_cast<microseconds>(add_points_stop - add_points_start).count()
    );
    comp.SaveRanking(ckp_root / "IDEC" / c.name / rank_ckp);
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
  fmt::print("Finished loading indices.\n");

  BatchMetric bm(args.batchsz, nb);

  for (auto efs : args.efs) {
    for (auto nrel : args.nrel) {
      time_t ts = time(nullptr);
      auto tm = localtime(&ts);
      std::string search_param = fmt::format("efs_{}_nrel_{}", efs, nrel);
      std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
      std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
      // fs::path log_root(fmt::format(LOGS, args.k) + "_special");
      fs::path log_root(fmt::format(LOGS, args.k));
      fs::path log_dir = log_root / method / workload / build_param / search_param;
      fs::create_directories(log_dir);
      fmt::print("Saving to {}.\n", (log_dir / out_json).string());
      FILE *out = stdout;
      nq = args.fast ? 1000 : nq;
#ifndef COMPASS_DEBUG
      fmt::print("Writing to {}.\n", (log_dir / out_text).string());
      out = fopen((log_dir / out_text).c_str(), "w");
#endif

      auto search_start = high_resolution_clock::system_clock::now();
#ifndef COMPASS_DEBUG
// #pragma omp parallel
// #pragma omp single
// #pragma omp taskloop
#endif
      for (int j = 0; j < nq; j += args.batchsz) {
        comp.SearchKnn(
            std::make_pair(xq + j * d, xqx + j * dx),
            args.batchsz,
            args.k,
            attrs.data(),
            &args.l_bound,
            &args.u_bound,
            efs,
            nrel,
            args.nthread,
            bm
        );
      }
      auto search_stop = high_resolution_clock::system_clock::now();
      auto search_time = duration_cast<microseconds>(search_stop - search_start).count();

      // statistics
      Stat stat(nq);
      for (int j = 0; j < nq; j += args.batchsz) {
        BatchMetric bm(args.batchsz, nb);
        auto search_start = high_resolution_clock::now();
        auto results = comp.SearchKnn(
            std::make_pair(xq + j * d, xqx + j * dx),
            args.batchsz,
            args.k,
            attrs.data(),
            &args.l_bound,
            &args.u_bound,
            efs,
            nrel,
            args.nthread,
            bm
        );
        auto search_stop = high_resolution_clock::now();
        bm.latency_in_us = duration_cast<microseconds>(search_stop - search_start).count();

        vector<float> gt_min_s(results.size()), gt_max_s(results.size());
        for (int i = 0; i < results.size(); i++) {
          gt_min_s[i] = dist_func(xq + (j + i) * d, xb + hybrid_topks[j + i].front() * d, &d);
          gt_max_s[i] = dist_func(xq + (j + i) * d, xb + hybrid_topks[j + i].back() * d, &d);
        }
        collect_batch_metric(results, bm, hybrid_topks, j, gt_min_s, gt_max_s, EPSILON, nsat, stat);
      }

      auto json = collate_stat(stat, nb, nsat, args.k, nq, search_time, args.nthread, out);
      std::ofstream ofs((log_dir / out_json).c_str());
      ofs.write(json.dump(4).c_str(), json.dump(4).length());
    }
  }
}