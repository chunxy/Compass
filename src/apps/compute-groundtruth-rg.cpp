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

    bool ok = true;
    for (int dim = 0; dim < da; dim++) {
      if (attrs[i][dim] < l_bounds[i * da + dim] || attrs[i][dim] > u_bounds[i * da + dim]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      auto dist = space.get_dist_func()(query, xb + i * d, space.get_dist_func_param());
      pq_topks[i].emplace(dist, i);
      if (pq_topks[i].size() > k) pq_topks[i].pop();
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
  configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  int nb = c.n_base, nq = c.n_queries, d = c.dim;

  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);

  l_bounds.resize(c.n_queries * c.attr_dim);
  u_bounds.resize(c.n_queries * c.attr_dim);
  auto rg = load_float32(c.attr_path, c.n_queries * 2, c.attr_dim);
  memcpy(l_bounds.data(), rg, c.n_queries * c.attr_dim * sizeof(float));
  memcpy(u_bounds.data(), rg + c.n_queries * c.attr_dim, c.n_queries * c.attr_dim * sizeof(float));

  if (l_bounds.size() != c.n_queries * c.attr_dim || u_bounds.size() != c.n_queries * c.attr_dim) {
    fmt::print("Attribute dimension mismatch: {} != {}\n", l_bounds.size(), c.n_queries * c.attr_dim);
    return 1;
  }

  vector<vector<pair<float, uint32_t>>> hybrid_topks(nq);
  int nsat;
  compute_groundtruth(xb, nb, xq, nq, d, c.attr_dim, l_bounds, u_bounds, attrs, k, nsat, hybrid_topks);

  std::string path = c.groundtruth_path;
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