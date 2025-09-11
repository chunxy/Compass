#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <boost/filesystem.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include "config.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  IvfGraph1dArgs args(argc, argv);

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

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt(c, {args.l_bound}, {args.u_bound}, args.k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");

  fs::path data("/home/chunxy/repos/Compass/data/navix");
  int sel = (args.u_bound - args.l_bound) / 100;
  fs::create_directories(data / c.name);
  std::string query_file = fmt::format("query_{}.txt", sel);
  std::string gt_file = fmt::format("gt_{}.txt", sel);
  std::ofstream query_ofs((data / c.name / query_file).string());
  std::ofstream gt_ofs((data / c.name / gt_file).string());

  for (int i = 0; i < 50; i++) {
    float *vector = xq + i * d;
    std::stringstream ss;
    ss << "[";
    for (int j = 0; j < d - 1; j++) {
      ss << std::fixed << std::setw(10) << std::setprecision(8) << vector[j] << ",";
      // ss << std::fixed << std::setprecision(8) << vector[j] << ",";
    }
    ss << std::fixed << std::setprecision(8) << vector[d - 1] << "]";
    std::string vector_string = ss.str();

    std::string query = fmt::format(
        "MATCH (e:{}) WHERE e.id >= {} AND e.id <= {} "
        "CALL ANN_SEARCH(e.embedding, {}, <maxK>, <efsearch>, <useQ>, "
        "<knn>, <searchType>) RETURN e.id;\n",
        c.name,
        args.l_bound,
        args.u_bound,
        vector_string
    );
    query_ofs << query;

    auto gt = hybrid_topks[i];
    for (int j = 0; j < gt.size(); j++) {
      int64_t id = gt[j];
      // write id to gt_ofs in binary
      gt_ofs.write((char *)&id, sizeof(id));
    }
  }
  query_ofs.flush();
  gt_ofs.flush();
}