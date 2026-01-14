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
#include "fc/btree.h"
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/predicate.h"

namespace fc = frozenca;
namespace fs = boost::filesystem;
using namespace std::chrono;
using std::array;
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

  std::string method = "Prefiltering";
  std::string workload = fmt::format(HYBRID_WORKLOAD_REVISION_TMPL, c.name, c.attr_range, args.k, c.type);

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

  // Load query range and groundtruth for hybrid search.
  args.l_bounds.resize(c.n_queries * c.attr_dim);
  args.u_bounds.resize(0);
  std::string rg_path = fmt::format(HYBRID_RG_REVISION_PATH_TMPL, c.name, c.attr_dim, c.attr_range, c.type);
  auto rg = load_float32(rg_path, c.n_queries, c.attr_dim);
  memcpy(args.l_bounds.data(), rg, c.n_queries * c.attr_dim * sizeof(float));

  if (c.attr_dim != 1) {
    throw std::runtime_error("Must be 1d negation\n");
    exit(1);
  }

  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt_revision(c, args.k, hybrid_topks);
  fmt::print("Finished loading query range and groundtruth.\n");

  // Compute selectivity.
  int nsat = 0;
  // stat_selectivity_revision(attrs, nb, c.attr_dim, args.l_bounds, args.u_bounds, nsat);

  vector<labeltype> labels(nb);
  std::iota(labels.begin(), labels.end(), 0);

  fc::BTreeMultiMap<float, labeltype> btree;
  for (int i = 0; i < nb; i++) {
    btree.insert(fc::BTreePair<float, labeltype>(std::move(attrs[i * c.attr_dim + 0]), std::move(labels[i])));
  }
  L2Space space(d);

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  // fs::path log_root(fmt::format(LOGS, args.k) + "_special");
  fs::path log_root(fmt::format(LOGS, args.k));
  fs::path log_dir = log_root / method / workload;
  fs::create_directories(log_dir);
  fmt::print("Saving to {}.\n", (log_dir / out_json).string());
  FILE *out = stdout;
#ifndef COMPASS_DEBUG
  fmt::print("Writing to {}.\n", (log_dir / out_text).string());
  out = fopen((log_dir / out_text).c_str(), "w");
#endif

  nq = args.fast ? 200 : nq;
  vector<priority_queue<pair<float, labeltype>>> results(nq);
  vector<int> num_computations(nq);
  vector<long> latencies(nq);
  long long total_search_time = 0;

#ifndef COMPASS_DEBUG
// #pragma omp parallel for num_threads(args.nthread) schedule(static)
#endif
  for (int j = 0; j < nq; j++) {
    auto search_start = high_resolution_clock::now();

    bool first_used_up = false;
    auto beg = btree.lower_bound(std::numeric_limits<float>::min());
    auto end = btree.upper_bound(args.l_bounds[j * c.attr_dim + 0] - 1e-5);
    num_computations[j] = 0;
    while (beg != end || !first_used_up) {
      if (beg == end) {
        first_used_up = true;
        beg = btree.lower_bound(args.l_bounds[j * c.attr_dim + 0] + 1e-5);
        end = btree.upper_bound(std::numeric_limits<float>::max());
      }
      int i = beg->second;
      beg++;
#ifdef USE_SSE
      if (beg != end) {
        _mm_prefetch(xb + (beg->second) * d, _MM_HINT_T0);
      }
#endif

      num_computations[j]++;
      results[j].push(std::make_pair(space.get_dist_func()(xq + j * d, xb + i * d, space.get_dist_func_param()), i));
      if (results[j].size() > args.k) results[j].pop();
    }
    auto search_stop = high_resolution_clock::now();
    auto search_time = duration_cast<microseconds>(search_stop - search_start).count();
    total_search_time += search_time;
    latencies[j] = search_time;
  }

  // statistics
  Stat stat(nq);
  for (int j = 0; j < nq;) {
    auto rz = results[j];
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
    stat.latencies.push_back(latencies[j]);
    j++;
  }

  auto json = collate_stat(stat, nb, nsat, args.k, nq, total_search_time, args.nthread, out);
  std::ofstream ofs((log_dir / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}