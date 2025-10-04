
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <algorithm>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "btree_map.h"
#include "config.h"
#include "utils/card.h"
#include "utils/funcs.h"

using std::pair;
using std::priority_queue;
using std::vector;

namespace po = boost::program_options;
using namespace std::chrono;

void compute_filter_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const vector<int> &blabels,
    const vector<int> &qlabels,
    const int k,
    int &nsat,
    vector<vector<pair<float, uint32_t>>> &hybrid_topks
) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  hybrid_topks.resize(nq);

  vector<vector<labeltype>> label_to_base_id;
  label_to_base_id.resize(1000);

  btree::btree_map<float, uint32_t> btree;
  for (int i = 0; i < nb; i++) {
    auto label = blabels[i];
    label_to_base_id[label].push_back(i);
  }

  hnswlib::L2Space space(d);
  omp_set_num_threads(omp_get_max_threads() - 4);
  auto compute_start = high_resolution_clock::now();
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;
    auto targets = label_to_base_id[qlabels[i]];
    for (int j = 0; j < std::min(targets.size(), (size_t)k); j++) {
      auto idx = targets[j];
      auto dist = space.get_dist_func()(query, xb + idx * d, &d);
      pq_topks[i].emplace(dist, idx);
    }
    for (int j = k; j < targets.size(); j++) {
      auto idx = targets[j];
      auto dist = space.get_dist_func()(query, xb + idx * d, &d);
      pq_topks[i].emplace(dist, idx);
      pq_topks[i].pop();
    }
    hybrid_topks[i].resize(pq_topks[i].size());
    int sz = pq_topks[i].size();
    while (pq_topks[i].size() != 0) {
      hybrid_topks[i][--sz] = pq_topks[i].top();
      pq_topks[i].pop();
    }
  }
  auto compute_end = high_resolution_clock::now();
  fmt::print(
      "Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count()
  );
}

int main(int argc, char **argv) {
  std::string dataname;
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
  vector<int> blabels;
  vector<int> qlabels;
  load_filter_data(c, xb, xq, gt, blabels, qlabels);

  vector<vector<pair<float, uint32_t>>> hybrid_topks(nq);
  int nsat;
  compute_filter_groundtruth(xb, nb, xq, nq, d, blabels, qlabels, k, nsat, hybrid_topks);

  std::string path = fmt::format(FILTER_GT_PATH_TMPL, c.name, c.attr_range, k);
  fmt::format("Saving to {}", path);
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