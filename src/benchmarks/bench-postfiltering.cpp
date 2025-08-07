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
#include <string>
#include <vector>
#include "config.h"
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/predicate.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using std::pair;
using std::vector;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  IvfGraph2dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries

  std::string method = "Postfiltering";
  std::string workload = fmt::format(
      HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, fmt::join(args.l_bounds, "-"), fmt::join(args.u_bounds, "-"), args.k
  );
  std::string build_param = fmt::format("M_{}_efc_{}", args.M, args.efc);

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

  L2Space space(d);
  HierarchicalNSW<float> comp(&space, nb, args.M, args.efc);
  fs::path ckp_root(CKPS);
  // std::string checkpoint = fmt::format(COMPASS_CHECKPOINT_TMPL, M, efc, nlist);
  std::string graph_ckp = fmt::format(COMPASS_GRAPH_CHECKPOINT_TMPL, args.M, args.efc);
  fs::path ckp_dir = ckp_root / "CompassR1d" / c.name;
  if (fs::exists(ckp_dir / graph_ckp)) {
    comp.loadIndex((ckp_dir / graph_ckp).string(), &space);
    fmt::print("Finished loading graph index.\n");
  } else {
    auto build_index_start = high_resolution_clock::now();
    for (int i = 0; i < nb; i++) comp.addPoint(xb + i * d, i);
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building graph, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    comp.saveIndex((ckp_dir / graph_ckp).string());
  }
  vector<labeltype> labels(nb);
  std::iota(labels.begin(), labels.end(), 0);
  fmt::print("Finished loading indices.\n");

  RangeQuery<float> pred(args.l_bounds.data(), args.u_bounds.data(), attrs, nb, c.attr_dim);
  // vector<QueryMetric> metrics(args.batchsz, QueryMetric(nb));

  for (auto efs : args.efs) {
    time_t ts = time(nullptr);
    auto tm = localtime(&ts);
    std::string search_param = fmt::format("efs_{}", efs);
    std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
    std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
    // fs::path log_root(fmt::format(LOGS, args.k) + "_special");
    fs::path log_root(fmt::format(LOGS, args.k));
    fs::path log_dir = log_root / method / workload / build_param / search_param;
    fs::create_directories(log_dir);
    fmt::print("Saving to {}.\n", (log_dir / out_json).string());
    FILE *out = stdout;
#ifndef COMPASS_DEBUG
    fmt::print("Writing to {}.\n", (log_dir / out_text).string());
    out = fopen((log_dir / out_text).c_str(), "w");
#endif

    comp.setEf(efs);
    vector<priority_queue<pair<float, labeltype>>> results(nq);
    vector<int> num_computations(nq);
    auto search_start = high_resolution_clock::now();
#ifndef COMPASS_DEBUG
// #pragma omp parallel for num_threads(args.nthread) schedule(static)
#endif
    for (int j = 0; j < nq; j++) {
      int initial_comp = comp.metric_distance_computations.load();
      auto ret = comp.searchKnn(xq + j * d, efs, nullptr);
      num_computations[j] = comp.metric_distance_computations.load() - initial_comp;
      while (!ret.empty()) {
        auto top = ret.top();
        if (pred(top.second)) {
          results[j].push(top);
          if (results[j].size() > args.k) results[j].pop();
        }
        ret.pop();
      }
    }
    auto search_stop = high_resolution_clock::now();
    auto search_time = duration_cast<microseconds>(search_stop - search_start).count();

    // statistics
    Stat stat(nq);
    for (int j = 0; j < nq;) {
      for (int ii = 0; ii < results.size(); ii++) {
        auto rz = results[ii];
        auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
        auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);
        float rz_min = std::numeric_limits<float>::max(), rz_max = std::numeric_limits<float>::min();
        int ivf_ppsl_in_rz = 0, graph_ppsl_in_rz = 0;
        int ivf_ppsl_in_tp = 0, graph_ppsl_in_tp = 0;
        while (!rz.empty()) {
          auto pair = rz.top();
          rz.pop();
          auto i = pair.second;
          auto d = pair.first;
          rz_min = std::min(rz_min, d);
          rz_max = std::max(rz_max, d);

          graph_ppsl_in_rz++;
          if (d <= gt_max + EPSILON) {
            graph_ppsl_in_tp++;
          }
        }

        stat.rz_min_s[j] = rz_min;
        stat.rz_max_s[j] = rz_max;
        stat.ivf_ppsl_in_rz_s[j] = ivf_ppsl_in_rz;
        stat.graph_ppsl_in_rz_s[j] = graph_ppsl_in_rz;
        stat.gt_min_s[j] = gt_min;
        stat.gt_max_s[j] = gt_max;
        stat.ivf_ppsl_in_tp_s[j] = ivf_ppsl_in_tp;
        stat.graph_ppsl_in_tp_s[j] = graph_ppsl_in_tp;

        stat.tp_s[j] = ivf_ppsl_in_tp + graph_ppsl_in_tp;
        stat.rz_s[j] = rz.size();
        stat.rec_at_ks[j] = (double)stat.tp_s[j] / hybrid_topks[j].size();
        stat.pre_at_ks[j] = (double)stat.tp_s[j] / rz.size();

        stat.ivf_ppsl_nums[j] = 0;
        stat.graph_ppsl_nums[j] = args.k;
        stat.ivf_ppsl_qlty[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_tp / stat.ivf_ppsl_nums[j] : 0;
        stat.ivf_ppsl_rate[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_rz / stat.ivf_ppsl_nums[j] : 0;
        stat.graph_ppsl_qlty[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_tp / stat.graph_ppsl_nums[j] : 0;
        stat.graph_ppsl_rate[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_rz / stat.graph_ppsl_nums[j] : 0;
        stat.perc_of_ivf_ppsl_in_tp[j] = stat.tp_s[j] != 0 ? (double)ivf_ppsl_in_tp / stat.tp_s[j] : 0;
        stat.perc_of_ivf_ppsl_in_rz[j] = (double)ivf_ppsl_in_rz / rz.size();
        stat.linear_scan_rate[j] = (double)stat.ivf_ppsl_nums[j] / nsat;
        stat.num_computations[j] = num_computations[j];
        stat.num_rounds[j] = 0;
        stat.latencies.push_back(duration_cast<microseconds>(search_stop - search_start).count());
        j++;
      }
    }

    auto json = collate_stat(stat, nb, nsat, args.k, nq, search_time, args.nthread, out);
    std::ofstream ofs((log_dir / out_json).c_str());
    ofs.write(json.dump(4).c_str(), json.dump(4).length());
  }
}