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

  args.k = 100;  // We find top-100 for Navix.

  fs::path data("/home/chunxy/repos/Compass/data/navix");
  int sel = (args.u_bound - args.l_bound) / 10000;
  fs::create_directories(data / c.name / "bench_data");
  std::string query_file = fmt::format("queries_{}.txt", sel);
  std::string gt_file = fmt::format("gt_{}.bin", sel);
  std::ofstream query_ofs((data / c.name / "bench_data" / query_file).string());
  std::ofstream gt_ofs((data / c.name / "bench_data" / gt_file).string());

  hnswlib::L2Space space(d);

  for (int i = 0; i < 50; i++) {
    float *query_vec = xq + i * d;
    std::stringstream ss;
    ss << "[";
    for (int j = 0; j < d - 1; j++) {
      ss << std::fixed << std::setw(10) << std::setprecision(8) << query_vec[j] << ",";
      // ss << std::fixed << std::setprecision(8) << vector[j] << ",";
    }
    ss << std::fixed << std::setprecision(8) << query_vec[d - 1] << "]";
    std::string vector_string = ss.str();

    std::string query = fmt::format(
        "MATCH (e:{}) WHERE e.id >= {} AND e.id < {} "
        "CALL ANN_SEARCH(e.embedding, {}, <maxK>, <efsearch>, <useQ>, "
        "<knn>, <searchType>) RETURN e.id;\n",
        c.name,
        args.l_bound,
        args.u_bound,
        vector_string
    );
    query_ofs << query;

    priority_queue<pair<float, uint32_t>> pq;
    for (int i = args.l_bound; i < args.u_bound; i++) {
      float dist = space.get_dist_func()(query_vec, xb + i * d, space.get_dist_func_param());
      pq.emplace(dist, i);
      if (pq.size() > args.k) {
        pq.pop();
      }
    }
    vector<labeltype> topk(args.k);
    int sz = args.k;
    while (!pq.empty()) {
      topk[--sz] = pq.top().second;
      pq.pop();
    }

    for (int j = 0; j < topk.size(); j++) {
      int64_t id = topk[j];
      // Write 8-byte id to gt_ofs in binary.
      gt_ofs.write((char *)&id, sizeof(id));
    }
  }
  query_ofs.flush();
  gt_ofs.flush();
}