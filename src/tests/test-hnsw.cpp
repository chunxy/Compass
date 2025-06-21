#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/reader.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using namespace hnswlib;
using std::vector;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  // IvfGraph1dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card["siftsmall_1_1000_top500_float32"];
  float l = 0, r = 1000;

  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  int M = 4, efc = 200;

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path root("/home/chunxy/repos/Compass/scratches/test-hnsw");

  fmt::print("Saving to {}.\n", (root / out_json).string());

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<uint32_t>> hybrid_topks(nq);
  int k = ng;
  IVecItrReader groundtruth_it(c.groundtruth_path);
  int i = 0;
  while (!groundtruth_it.HasEnded()) {
    auto next = groundtruth_it.Next();
    if (next.size() != ng) {
      throw fmt::format("ng ({}) is greater than the size of the groundtruth ({})", ng, next.size());
    }
    hybrid_topks[i].resize(k);
    memcpy(hybrid_topks[i].data(), next.data(), k * sizeof(uint32_t));
    i++;
  }
  fmt::print("Finished loading groundtruth.\n");

  L2Space l2space(d);
  HierarchicalNSW<float> *comp;

  string index_file = fmt::format("{}_M_{}_efc_{}.hnsw", c.name, M, efc);
  fs::path ckp_path = root / index_file;
  if (fs::exists(ckp_path)) {
    comp = new HierarchicalNSW<float>(&l2space, ckp_path.string(), false, nb);
  } else {
    comp = new HierarchicalNSW<float>(&l2space, nb, M, efc);
    auto build_index_start = high_resolution_clock::now();
    for (int i = 0; i < nb; i++) comp->addPoint(xb + i * d, i);
    auto build_index_stop = high_resolution_clock::now();
    comp->saveIndex(ckp_path.string());
  }
  fmt::print("Finished loading/building index\n");

  nlohmann::json json;
  for (auto efs : {50, 100, 150, 200}) {
    int initial_ncomp = comp->metric_distance_computations.load();
    int initial_nhops = comp->metric_hops.load();
    comp->setEf(efs);

    auto search_start = high_resolution_clock::system_clock::now();
    for (int j = 0; j < nq; j++) {
      auto results = comp->searchKnn(xq + j * d, k);
    }
    auto search_stop = high_resolution_clock::system_clock::now();
    auto search_time = duration_cast<microseconds>(search_stop - search_start).count();

    int tp = 0;
    for (int j = 0; j < nq; j++) {
      auto results = comp->searchKnn(xq + j * d, k);

      auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
      auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);
      while (results.size()) {
        auto pair = results.top();
        results.pop();
        auto i = pair.second;
        auto d = pair.first;
        if (d <= gt_max + 1e-5) tp++;
      }
    }
    json[fmt::to_string(efs)]["recall"] = (double)tp / nq / k;
    json[fmt::to_string(efs)]["qps"] = nq * 1000000. / search_time;
    json[fmt::to_string(efs)]["num_computations"] =
        (comp->metric_distance_computations.load() - initial_ncomp) / 2. / nq;
    json[fmt::to_string(efs)]["nhops"] = (comp->metric_hops.load() - initial_nhops) / 2. / nq;
  }

  std::ofstream ofs((root / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}