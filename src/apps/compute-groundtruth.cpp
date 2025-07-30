#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <omp.h>
#include <boost/core/no_exceptions_support.hpp>
#include <boost/geometry.hpp>
#include <boost/program_options.hpp>
#include <chrono>
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

void compute_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const float l_bound,
    const float u_bound,
    const vector<float> &attrs,
    const int k,
    int &nsat,
    vector<vector<pair<float, uint32_t>>> &hybrid_topks
) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  hybrid_topks.resize(nq);

  btree::btree_map<float, uint32_t> btree;
  auto btree_start = high_resolution_clock::now();
  for (int i = 0; i < attrs.size(); i++) {
    btree.insert(pair<float, uint32_t>{attrs[i], i});
  }
  auto btree_stop = high_resolution_clock::now();

  fmt::print(
      "B-tree construction and search took {} microseconds\n",
      duration_cast<microseconds>(btree_stop - btree_start).count()
  );

  auto beg = btree.lower_bound(l_bound);
  auto end = btree.upper_bound(u_bound);
  auto targets = vector<pair<float, uint32_t>>(beg, end);

  hnswlib::L2Space space(d);
  auto compute_start = high_resolution_clock::now();
#pragma omp for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;

    for (int j = 0; j < targets.size(); j++) {
      auto idx = targets[j].second;
      auto dist = space.get_dist_func()(query, xb + idx * d, &d);
      pq_topks[i].emplace(dist, idx);
    }
    while (pq_topks[i].size() > k) pq_topks[i].pop();
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

void compute_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const vector<vector<float>> &attrs,
    const int k,
    int &nsat,
    vector<vector<pair<float, uint32_t>>> &hybrid_topks
) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  hybrid_topks.resize(nq);
  namespace geo = boost::geometry;
  using point = geo::model::point<double, 2, geo::cs::cartesian>;
  using box = geo::model::box<point>;
  using value = std::pair<point, unsigned>;
  using rtree = geo::index::rtree<value, geo::index::quadratic<16>>;

  rtree rt;
  auto rtree_start = high_resolution_clock::now();
  for (int i = 0; i < attrs.size(); i++) {
    rt.insert(value({point(attrs[i][0], attrs[i][1]), i}));
  }
  auto rtree_stop = high_resolution_clock::now();
  fmt::print(
      "R-tree construction took {} microseconds\n", duration_cast<microseconds>(rtree_stop - rtree_start).count()
  );

  box b(point(l_bounds[0], l_bounds[1]), point(u_bounds[0], u_bounds[1]));
  auto beg = rt.qbegin(geo::index::covered_by(b));
  auto end = rt.qend();
  auto targets = vector<value>(beg, end);

  hnswlib::L2Space space(d);
  auto compute_start = high_resolution_clock::now();
#pragma omp for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;

    for (int j = 0; j < k; j++) {
      auto idx = targets[j].second;
      auto dist = space.get_dist_func()(query, xb + idx * d, &d);
      pq_topks[i].emplace(dist, idx);
    }
    for (int j = k; j < targets.size(); j++) {
      auto idx = targets[j].second;
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
  fmt::print("Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count());
}

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

  vector<btree::btree_map<float, uint32_t>> btrees(da);
  auto btrees_start = high_resolution_clock::now();
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < da; j++) {
      btrees[j].insert(pair<float, uint32_t>{attrs[i][j], i});
    }
  }
  auto btrees_stop = high_resolution_clock::now();
  fmt::print(
      "B-trees construction took {} microseconds\n", duration_cast<microseconds>(btrees_stop - btrees_start).count()
  );

  std::vector<std::unordered_set<uint32_t>> candidates_per_dim(da);
  for (int j = 0; j < da; ++j) {
    auto beg = btrees[j].lower_bound(l_bounds[j]);
    auto end = btrees[j].upper_bound(u_bounds[j]);
    for (auto itr = beg; itr != end; ++itr) {
      candidates_per_dim[j].insert(itr->second);
    }
  }
  // Intersect all sets in candidates_per_dim
  std::unordered_set<uint32_t> intersection;
  for (auto c : candidates_per_dim[0]) {
    intersection.insert(c);
  }
  for (int j = 1; j < da; ++j) {
    std::unordered_set<uint32_t> temp;
    for (const auto &id : intersection) {
      if (candidates_per_dim[j].count(id)) {
        temp.insert(id);
      }
    }
    intersection = std::move(temp);
  }
  vector<uint32_t> targets(intersection.begin(), intersection.end());

  hnswlib::L2Space space(d);
  auto compute_start = high_resolution_clock::now();
#pragma omp for schedule(static)
  for (int i = 0; i < nq; i++) {
    const float *query = xq + i * d;

    for (int j = 0; j < k; j++) {
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
  fmt::print("Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count());
}

void compute_groundtruth(
    const float *xb,
    const int nb,
    const float *xq,
    const int nq,
    const size_t d,
    const size_t da,
    const float *l_bounds,
    const float *u_bounds,
    const vector<vector<float>> &attrs,
    const int k,
    int &nsat,
    vector<vector<pair<float, uint32_t>>> &hybrid_topks
) {
  vector<priority_queue<pair<float, uint32_t>>> pq_topks(nq);
  hybrid_topks.resize(nq);

  vector<btree::btree_map<float, uint32_t>> btrees(da);
  auto btrees_start = high_resolution_clock::now();
  for (int i = 0; i < nb; i++) {
    for (int j = 0; j < da; j++) {
      btrees[j].insert(pair<float, uint32_t>{attrs[i][j], i});
    }
  }
  auto btrees_stop = high_resolution_clock::now();
  fmt::print(
      "B-trees construction took {} microseconds\n", duration_cast<microseconds>(btrees_stop - btrees_start).count()
  );

  hnswlib::L2Space space(d);
  auto compute_start = high_resolution_clock::now();
#pragma omp for schedule(static)
  for (int i = 0; i < nq; i++) {
    vector<uint32_t> targets;
    auto beg = btrees[0].lower_bound(l_bounds[i * da]);
    auto end = btrees[0].upper_bound(u_bounds[i * da]);
    for (auto itr = beg; itr != end; ++itr) {
      int idx = itr->second, j = 1;
      while (j < da) {
        if (attrs[idx][j] < l_bounds[i * da + j] || attrs[idx][j] > u_bounds[i * da + j]) {
          break;
        }
        j++;
      }
      if (j == da) {
        targets.push_back(idx);
      }
    }

    const float *query = xq + i * d;
    for (int j = 0; j < k; j++) {
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
  fmt::print("Computation took {} microseconds\n", duration_cast<microseconds>(compute_end - compute_start).count());
}

int main(int argc, char **argv) {
  std::string dataname;
  vector<float> l_bounds, u_bounds;
  vector<int> percents;

  uint32_t k;

  po::options_description configs;
  configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  configs.add_options()("p", po::value<decltype(percents)>(&percents)->required()->multitoken()->required());
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
  std::mt19937 rng;
  rng.seed(0);
  std::uniform_real_distribution<float> distrib_real;
  for (size_t i = 0; i < nq; i++) {
    for (size_t j = 0; j < c.attr_dim; j++) {
      float num = (distrib_real(rng) * (float)c.attr_range * (100 - percents[j]) / 100);
      l_bounds.push_back(num);
      u_bounds.push_back(num + (float)c.attr_range * percents[j] / 100);
    }
  }
  vector<vector<pair<float, uint32_t>>> hybrid_topks(nq);
  int nsat;
  compute_groundtruth(xb, nb, xq, nq, d, c.attr_dim, l_bounds.data(), u_bounds.data(), attrs, k, nsat, hybrid_topks);

  std::string gt_path = fmt::format(HYBRID_GT_PATH_TMPL_NEO, c.name, c.attr_range, percents, k);
  std::ofstream gt_ofs(gt_path, std::ios_base::binary & std::ios_base::out);
  for (int i = 0; i < nq; i++) {
    uint32_t size = hybrid_topks[i].size();
    gt_ofs.write((char *)&size, sizeof(size));
    for (int j = 0; j < hybrid_topks[i].size(); j++) {
      gt_ofs.write((char *)&hybrid_topks[i][j].second, 4);
    }
  }

  std::string rg_path = fmt::format(HYBRID_RG_PATH_TMPL, c.name, c.attr_range, percents);
  std::ofstream rg_ofs(rg_path, std::ios_base::binary & std::ios_base::out);
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < c.attr_dim; j++) {
      rg_ofs.write((char *)&l_bounds[i * c.attr_dim + j], sizeof(float));
    }
  }
  for (int i = 0; i < nq; i++) {
    for (int j = 0; j < c.attr_dim; j++) {
      rg_ofs.write((char *)&u_bounds[i * c.attr_dim + j], sizeof(float));
    }
  }

  return 0;
}