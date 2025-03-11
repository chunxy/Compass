#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "config.h"
#include "methods/acorn/AcornUtils.h"
#include "methods/acorn/IndexACORN.h"
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

  int M = 32;       // Tightly connected with internal dimensionality of
                    // the data and memory consumption.
  int M_beta = 64;  // param for compression
  int efc = 200;    // Controls index search speed/build speed tradeoff; 40 by default.
  int gamma = 12;
  int efs = 100;  // default is 16
  // float attr_sel = 0.001;
  // int gamma = (int) 1 / attr_sel;
  // int n_centroids;
  std::string assignment_type = "rand";

  po::options_description configs;
  po::options_description required_configs("Required");
  // dataset parameter
  required_configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  // search parameters
  required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  po::options_description optional_configs("Optional");
  // index construction parameters
  optional_configs.add_options()("M", po::value<decltype(M)>(&M));
  optional_configs.add_options()("beta", po::value<decltype(M_beta)>(&M_beta));
  optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
  optional_configs.add_options()("gamma", po::value<decltype(gamma)>(&gamma));
  // index search parameters
  optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs));

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

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string method = "acorn";
  std::string workload = fmt::format(FILTER_WORKLOAD_TMPL, c.name, c.attr_range, k);
  std::string param = fmt::format("M_{}_beta_{}_efc_{}_gamma_{}_efs_{}", M, M_beta, efc, gamma, efs);
  std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path log_root(fmt::format(LOGS, args.k));
  fs::path log_dir = log_root / method / workload / param;
  fs::create_directories(log_dir);

  fmt::print("Saving to {}.\n", (log_dir / out_json).string());
  fmt::print("Writing to {}.\n", (log_dir / out_text).string());
  // #ifndef NDEBUG
  freopen((log_dir / out_text).c_str(), "w", stdout);
  // #endif

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  vector<int> blabels;
  vector<int> qlabels;
  load_filter_data(c, xb, xq, gt, blabels, qlabels);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks;
  load_filter_query_gt(c, k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");
  // Compute selectivity.
  int nsat = (1. / c.attr_range) * nb;
  // stat_selectivity(labels, l_bound, u_bound, nsat);

  acorn::IndexACORNFlat *hybrid_index;
  fs::path ckp_root(CKPS);
  std::string acorn_ckp = fmt::format(ACORN_CHECKPOINT_TMPL, M, M_beta, efc, gamma);
  fs::path ckp_dir = ckp_root / method / c.name;
  auto ckp_path = ckp_dir / acorn_ckp;
  if (fs::exists(ckp_dir / acorn_ckp)) {
    hybrid_index = dynamic_cast<acorn::IndexACORNFlat *>(acorn::read_acorn_index(ckp_path.c_str()));
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
  hybrid_index->acorn.efSearch = efs;

  // search
  vector<faiss::idx_t> nn(k * nq);
  vector<float> dist(k * nq);
  // create filter_ids_map, ie a bitmap of the ids that are in the filter

  int batch_size = 1'000;
  int n_batches = (nq + (batch_size - 1)) / batch_size;
  vector<vector<char>> filter_id_maps(n_batches);
  for (int b = 0; b < n_batches; b++) {
    filter_id_maps[b].resize(batch_size * nb);
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < nb; j++) {
        filter_id_maps[b][i * nb + j] = (bool)(blabels[j] == qlabels[b * batch_size + i]);
      }
    }
  }
  filter_id_maps[n_batches - 1].resize((nq % batch_size) * nb);
  for (int i = 0; i < (nq % batch_size); i++) {
    for (int j = 0; j < nb; j++) {
      filter_id_maps[n_batches - 1][i * nb + j] =
          (bool)(blabels[j] == qlabels[(n_batches - 1) * batch_size + i]);
    }
  }

  auto search_start = high_resolution_clock::now();
  for (int i = 0; i < n_batches - 1; i++) {
    hybrid_index->search(
        batch_size,
        xq + i * batch_size * d,
        k,
        dist.data() + i * batch_size * k,
        nn.data() + i * batch_size * k,
        filter_id_maps[i].data()
    );
  }
  hybrid_index->search(
      batch_size,
      xq + (n_batches - 1) * batch_size * d,
      k,
      dist.data() + (n_batches - 1) * batch_size * k,
      nn.data() + (n_batches - 1) * batch_size * k,
      filter_id_maps[n_batches - 1].data()
  );
  // hybrid_index->search(nq, xq, k, dist.data(), nn.data(), filter_id_maps.data());
  auto search_stop = high_resolution_clock::now();
  auto search_time = duration_cast<milliseconds>(search_stop - search_start).count();

  // statistics
  vector<float> rec_at_ks(nq);
  vector<float> pre_at_ks(nq);
  for (int j = 0; j < nq; j++) {
    fmt::print("Query: {:d},\n", j);
    std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;

    fmt::print("\tResult      : ");
    for (int i = 0; i < k; i++) {
      auto idx = nn[j * k + i];
      auto d = dist[j * k + i];
      if (rz_indices.find(idx) != rz_indices.end()) {
        fmt::print("Found duplicate item {:d} in result indices.\n", idx);
      }
      rz_indices.insert(idx);
    }
    fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", dist.front(), dist.back());

    fmt::print("\tGround Truth: ");
    int ivf_ppsl_in_tp = 0;
    for (auto i : hybrid_topks[j]) {
      gt_indices.insert(i);
    }
    auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
    auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);
    fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", gt_min, gt_max);

    std::set_intersection(
        gt_indices.begin(),
        gt_indices.end(),
        rz_indices.begin(),
        rz_indices.end(),
        std::inserter(rz_gt_interse, rz_gt_interse.begin())
    );
    rec_at_ks[j] = (float)rz_gt_interse.size() / gt_indices.size();
    pre_at_ks[j] = (float)rz_gt_interse.size() / rz_indices.size();
    fmt::print("\tRecall: {:5.2f}%, ", rec_at_ks[j] * 100);
    fmt::print("Precision: {:5.2f}%, ", pre_at_ks[j] * 100);
    fmt::print("{:3d}/{:3d}/{:3d}\n", rz_gt_interse.size(), rz_indices.size(), k);
  }

  fmt::print("Selectivity       : {}/{} = {:5.2f}%\n", nsat, nb, (double)nsat / nb * 100);
  collate_acorn_stats(search_time, rec_at_ks, pre_at_ks, (log_dir / out_json).string());
}