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
#include <string>
#include <vector>
#include "acorn/AcornUtils.h"
#include "acorn/IndexACORN.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/funcs.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std::chrono;
using std::vector;

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

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

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
  hybrid_index = new acorn::IndexACORNFlat(d, M, gamma, blabels, M_beta);
  auto build_index_start = high_resolution_clock::now();
  hybrid_index->add(nb, xb);
  auto build_index_stop = high_resolution_clock::now();
  fmt::print(
      "Finished building ACORN, took {} microseconds.\n",
      duration_cast<microseconds>(build_index_stop - build_index_start).count()
  );

  nlohmann::json json;
  json["build_time_in_s"] = duration_cast<milliseconds>(build_index_stop - build_index_start).count() / 1000.0;
  fs::path build_time_path =
      fs::path("/opt/nfs_dcc/chunxy/logs_10") / fmt::format("acorn_{}_build_time_in_seconds.json", c.name);
  std::ofstream ofs(build_time_path.string());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
  ofs.close();
}