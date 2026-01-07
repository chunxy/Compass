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

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

  CompassPostK<float, float> comp(
      nb, d, c.attr_dim, args.M, args.efc, args.nlist, args.M_cg, args.batch_k, args.initial_efs, args.delta_efs
  );

  std::vector<labeltype> labels(nb);
  std::iota(labels.begin(), labels.end(), 0);

  auto train_ivf_start = high_resolution_clock::now();
  comp.TrainIvf(nb, xb);
  auto train_ivf_stop = high_resolution_clock::now();
  fmt::print(
      "Finished training IVF, took {} microseconds.\n",
      duration_cast<microseconds>(train_ivf_stop - train_ivf_start).count()
  );

  auto add_points_start = high_resolution_clock::now();
  comp.AddPointsToIvf(nb, xb, labels.data(), attrs);
  auto add_points_stop = high_resolution_clock::now();
  fmt::print(
      "Finished adding points, took {} microseconds.\n",
      duration_cast<microseconds>(add_points_stop - add_points_start).count()
  );

  auto build_index_start = high_resolution_clock::now();
  comp.AddPointsToGraph(nb, xb, labels.data());
  auto build_index_stop = high_resolution_clock::now();
  fmt::print(
      "Finished building graph, took {} microseconds.\n",
      duration_cast<microseconds>(build_index_stop - build_index_start).count()
  );

  auto build_cgraph_start = high_resolution_clock::now();
  comp.BuildClusterGraph();
  auto build_cgraph_stop = high_resolution_clock::now();
  fmt::print(
      "Finished building cluster graph, took {} microseconds.\n",
      duration_cast<microseconds>(build_cgraph_stop - build_cgraph_start).count()
  );

  nlohmann::json json;
  json["train_ivf_time_in_s"] = duration_cast<milliseconds>(train_ivf_stop - train_ivf_start).count() / 1000.0;
  json["add_points_time_in_s"] = duration_cast<milliseconds>(add_points_stop - add_points_start).count() / 1000.0;
  json["build_index_time_in_s"] = duration_cast<milliseconds>(build_index_stop - build_index_start).count() / 1000.0;
  json["build_cgraph_time_in_s"] = duration_cast<milliseconds>(build_cgraph_stop - build_cgraph_start).count() / 1000.0;
  fs::path build_time_path =
      fs::path("/opt/nfs_dcc/chunxy/logs_10") / fmt::format("compass_{}_build_time_in_seconds.json", c.name);
  std::ofstream ofs(build_time_path.string());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
  ofs.close();
}