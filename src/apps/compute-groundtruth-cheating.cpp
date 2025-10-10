#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <omp.h>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <ios>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "config.h"
#include "utils/card.h"
#include "utils/funcs.h"

using std::pair;
using std::priority_queue;
using std::vector;

namespace po = boost::program_options;
using namespace std::chrono;

void compute_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const size_t da,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const vector<vector<float>> &attrs,
    const int k,
    int &nsat,
    vector<vector<pair<float, uint32_t>>> &hybrid_topks
) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  hybrid_topks.resize(nq);
  hnswlib::L2Space space(d);

  auto compute_start = high_resolution_clock::now();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;

    for (int j = l_bounds[0]; j <= u_bounds[0]; j++) {
      bool ok = true;
      for (int dim = 1; dim < da; dim++) {
        if (attrs[j][dim] < l_bounds[dim] || attrs[j][dim] > u_bounds[dim]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        auto dist = space.get_dist_func()(query, xb + j * d, space.get_dist_func_param());
        pq_topks[i].emplace(dist, j);
        if (pq_topks[i].size() > k) pq_topks[i].pop();
      }
    }
    hybrid_topks[i].resize(pq_topks[i].size());
    int sz = pq_topks[i].size();
    while (pq_topks[i].size() != 0) {
      hybrid_topks[i][--sz] = pq_topks[i].top();
      pq_topks[i].pop();
    }
  }
  auto compute_end = high_resolution_clock::now();
  fmt::print("Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count());
}

int main(int argc, char **argv) {
  std::string dataname;
  vector<float> l_bounds, u_bounds;

  uint32_t k;

  po::options_description configs;
  configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  configs.add_options()("l", po::value<decltype(l_bounds)>(&l_bounds)->required()->multitoken());
  configs.add_options()("r", po::value<decltype(u_bounds)>(&u_bounds)->required()->multitoken());
  configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  int nb = c.n_base, nq = c.n_queries, d = c.dim;

  if (c.attr_dim != 1) {
    fmt::print("Handling 1D case only for now.\n");
    return -1;
  }

  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);

  vector<vector<pair<float, uint32_t>>> hybrid_topks(nq);
  int nsat;
  compute_groundtruth(xb, nb, xq, nq, d, c.attr_dim, l_bounds, u_bounds, attrs, k, nsat, hybrid_topks);

  std::string path = fmt::format(HYBRID_GT_CHEATING_PATH_TMPL, c.name, l_bounds, u_bounds, k);
  fmt::print("Saving to {}\n", path);
  std::ofstream ofs(path, std::ios_base::binary & std::ios_base::out);
  for (int i = 0; i < nq; i++) {
    uint32_t size = hybrid_topks[i].size();
    ofs.write((char *)&size, sizeof(size));
    for (int j = 0; j < hybrid_topks[i].size(); j++) {
      ofs.write((char *)&hybrid_topks[i][j].second, 4);
    }
  }

  return 0;
}