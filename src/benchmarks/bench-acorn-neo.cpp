#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "acorn/AcornUtils.h"
#include "acorn/IndexACORN.h"
#include "config.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std::chrono;
using std::vector;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  std::string dataname;
  int k;
  vector<float> l_bounds, u_bounds;
  int M = 32;
  int M_beta = 64;  // param for compression
  int gamma = 12;
  vector<int> efs_s = {100};  // default is 16
  // float attr_sel = 0.001;
  // int gamma = (int) 1 / attr_sel;
  std::string assignment_type = "rand";
  bool fast = true;

  po::options_description configs;
  po::options_description required_configs("Required"), optional_configs("Optional");
  // workload parameters
  required_configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  required_configs.add_options()("l", po::value<decltype(l_bounds)>(&l_bounds)->multitoken()->required());
  required_configs.add_options()("r", po::value<decltype(u_bounds)>(&u_bounds)->multitoken()->required());
  // index construction parameters
  optional_configs.add_options()("M", po::value<decltype(M)>(&M));
  optional_configs.add_options()("beta", po::value<decltype(M_beta)>(&M_beta));
  optional_configs.add_options()("gamma", po::value<decltype(gamma)>(&gamma));
  // index search parameters
  optional_configs.add_options()("efs", po::value<decltype(efs_s)>(&efs_s)->multitoken());
  optional_configs.add_options()("fast", po::value<decltype(fast)>(&fast));

  // Merge required and optional configs.
  configs.add(required_configs).add(optional_configs);
  // Parse arguments.
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  size_t nb = c.n_base;      // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries

  std::string method = "ACORN";

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt(c, l_bounds, u_bounds, k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");
  // Compute selectivity.
  int nsat;
  stat_selectivity(attrs, nb, c.attr_dim, l_bounds, u_bounds, nsat);

  vector<int> blabels(nb);
  for (int i = 0; i < nb; i++) {
    bool ok = true;
    for (int j = 0; j < c.attr_dim; j++) {
      if (attrs[i * c.attr_dim + j] < l_bounds[j] || attrs[i * c.attr_dim + j] > u_bounds[j]) {
        ok = false;
        break;
      }
    }
    blabels[i] = (int)ok;
  }
  acorn::IndexACORNFlat *hybrid_index;
  fs::path ckp_root(CKPS);
  std::string acorn_ckp = fmt::format(ACORN_CHECKPOINT_TMPL, M, M_beta, gamma);
  fs::path ckp_dir = ckp_root / method / c.name;
  auto ckp_path = ckp_dir / acorn_ckp;
  if (fs::exists(ckp_dir / acorn_ckp)) {
    hybrid_index = dynamic_cast<acorn::IndexACORNFlat *>(acorn::read_acorn_index(ckp_path.c_str()));
    hybrid_index->acorn.metadata = blabels.data();
    fmt::print("Finished loading ACORN index.\n");
  } else {
    hybrid_index = new acorn::IndexACORNFlat(d, M, gamma, blabels, M_beta);
    auto build_index_start = high_resolution_clock::now();
    hybrid_index->add(nb, xb);
    auto build_index_stop = high_resolution_clock::now();
    fmt::print(
        "Finished building ACORN, took {} microseconds.\n",
        duration_cast<microseconds>(build_index_stop - build_index_start).count()
    );
    fs::create_directories(ckp_path.parent_path());
    acorn::write_acorn_index(hybrid_index, ckp_path.c_str());
  }

  for (auto efs : efs_s) {
    int initial_n3 = acorn::acorn_stats.n3;
    time_t ts = time(nullptr);
    auto tm = localtime(&ts);
    std::string workload =
        fmt::format(HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, fmt::join(l_bounds, "-"), fmt::join(u_bounds, "-"), k);
    std::string build = fmt::format("M_{}_beta_{}_gamma_{}", M, M_beta, gamma);
    std::string search = fmt::format("efs_{}", efs);
    std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
    std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
    // fs::path log_root(fmt::format(LOGS, k) + "_special");
    fs::path log_root(fmt::format(LOGS, k));
    fs::path log_dir = log_root / method / workload / build / search;

    hybrid_index->acorn.efSearch = efs;

    fs::create_directories(log_dir);
    fmt::print("Saving to {}.\n", (log_dir / out_json).string());
    FILE *out = stdout;
    nq = fast ? 200 : nq;
#ifndef COMPASS_DEBUG
    fmt::print("Writing to {}.\n", (log_dir / out_text).string());
    out = fopen((log_dir / out_text).c_str(), "w");
#endif

    // search
    vector<faiss::idx_t> nn(k * nq);
    vector<float> dist(k * nq);
    omp_set_num_threads(1);

    // Create filter_ids_map, i.e. a bitmap of the ids that are in the filter.
    int batch_size = 100;
    if (nq % batch_size != 0) {
      fmt::print("Warning: nq % batch_size != 0, will not be able to search in batches.\n");
      return -1;
    }

    // auto search_start = high_resolution_clock::now();
    // int n_batches = (nq + (batch_size - 1)) / batch_size;
    // vector<vector<char>> filter_id_maps(n_batches);
    // for (int b = 0; b < n_batches; b++) {
    //   filter_id_maps[b].resize(batch_size * nb);
    //   for (int i = 0; i < batch_size; i++) {
    //     for (int j = 0; j < nb; j++) {
    //       filter_id_maps[b][i * nb + j] = (bool)(blabels[j] == 1);
    //     }
    //   }
    // }

    auto search_start = high_resolution_clock::now();
    int n_batches = (nq + (batch_size - 1)) / batch_size;
    vector<char> filter_id_map(batch_size * nb);

    for (int i = 0; i < n_batches; i++) {
      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < nb; j++) {
          filter_id_map[i * nb + j] = (bool)(blabels[j] == 1);
        }
      }
      hybrid_index->search(
          batch_size,
          xq + i * batch_size * d,
          k,
          dist.data() + i * batch_size * k,
          nn.data() + i * batch_size * k,
          filter_id_map.data()
      );
    }
    auto search_stop = high_resolution_clock::now();
    auto search_time = duration_cast<milliseconds>(search_stop - search_start).count();

    // statistics
    vector<float> rec_at_ks(nq);
    vector<float> pre_at_ks(nq);
    for (int j = 0; j < nq; j++) {
      fmt::print(out, "Query: {:d},\n", j);
      std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;

      for (int i = 0; i < k; i++) {
        auto idx = nn[j * k + i];
        auto d = dist[j * k + i];
        if (rz_indices.find(idx) != rz_indices.end()) {
          fmt::print(out, "Found duplicate item {:d} in result indices.\n", idx);
        } else {
          rz_indices.insert(idx);
        }
      }
      fmt::print(out, "\tResult      : ");
      fmt::print(out, "Min: {:9.2f}, Max: {:9.2f}\n", dist[j * k], dist[j * k + k - 1]);

      fmt::print(out, "\tGround Truth: ");
      int ivf_ppsl_in_tp = 0;
      for (auto i : hybrid_topks[j]) {
        gt_indices.insert(i);
      }
      auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
      auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);
      fmt::print(out, "Min: {:9.2f}, Max: {:9.2f}\n", gt_min, gt_max);

      std::set_intersection(
          gt_indices.begin(),
          gt_indices.end(),
          rz_indices.begin(),
          rz_indices.end(),
          std::inserter(rz_gt_interse, rz_gt_interse.begin())
      );
      rec_at_ks[j] = (float)rz_gt_interse.size() / gt_indices.size();
      pre_at_ks[j] = (float)rz_gt_interse.size() / rz_indices.size();
      fmt::print(out, "\tRecall: {:5.2f}%, ", rec_at_ks[j] * 100);
      fmt::print(out, "Precision: {:5.2f}%, ", pre_at_ks[j] * 100);
      fmt::print(out, "{:3d}/{:3d}/{:3d}\n", rz_gt_interse.size(), rz_indices.size(), k);
    }

    fmt::print(out, "Selectivity       : {}/{} = {:5.2f}%\n", nsat, nb, (double)nsat / nb * 100);
    collate_acorn_stats(
        search_time, acorn::acorn_stats.n3 - initial_n3, rec_at_ks, pre_at_ks, (log_dir / out_json).string(), out
    );
  }
}