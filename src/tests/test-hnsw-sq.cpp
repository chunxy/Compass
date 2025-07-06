#include <faiss/IndexScalarQuantizer.h>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <queue>
#include <set>

#include "config.h"
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/reader.h"

using idx_t = faiss::idx_t;
namespace fs = std::filesystem;

int main() {
  auto dist_func = hnswlib::L2Sqr;
  extern DataCard siftsmall_1_1000_top500_float32;
  size_t d = siftsmall_1_1000_top500_float32.dim;      // dimension
  int nb = siftsmall_1_1000_top500_float32.n_base;     // database size
  int nq = siftsmall_1_1000_top500_float32.n_queries;  // nb of queries
  int nk = siftsmall_1_1000_top500_float32.n_groundtruth;

  std::string groundtruth_path = siftsmall_1_1000_top500_float32.groundtruth_path;
  std::string query_path = siftsmall_1_1000_top500_float32.query_path;
  std::string base_path = siftsmall_1_1000_top500_float32.base_path;
  std::string attr_path = fmt::format(
      VALUE_PATH_TMPL,
      siftsmall_1_1000_top500_float32.name,
      siftsmall_1_1000_top500_float32.attr_dim,
      siftsmall_1_1000_top500_float32.attr_range
  );

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path root("/home/chunxy/repos/Compass/scratches/test-hnsw-sq");
  fmt::print("Saving to {}.\n", (root / out_json).string());

  FVecItrReader base_it(base_path);
  FVecItrReader query_it(query_path);
  IVecItrReader groundtruth_it(groundtruth_path);
  float *xb = new float[d * nb];
  float *xq = new float[d * nq];
  uint32_t *gt = new uint32_t[nk * nq];
  int i = 0;
  while (!base_it.HasEnded()) {
    auto next = base_it.Next();
    memcpy(xb + i * d, next.data(), d * sizeof(float));
    i++;
  }
  i = 0;
  while (!query_it.HasEnded()) {
    auto next = query_it.Next();
    memcpy(xq + i * d, next.data(), d * sizeof(float));
    i++;
  }
  i = 0;
  while (!groundtruth_it.HasEnded()) {
    auto next = groundtruth_it.Next();
    memcpy(gt + i * nk, next.data(), nk * sizeof(uint32_t));
    i++;
  }
  AttrReaderToVector<float> reader(attr_path);
  auto attrs = reader.GetAttrs();

  faiss::IndexScalarQuantizer sq(d, faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform, faiss::METRIC_L2);
  sq.train(nb, xb);
  sq.add(nb, xb);

  int M = 4;
  hnswlib::HierarchicalNSW<int> hnsw(new hnswlib::L2SpaceB(d), nb, M, 200);
  uint8_t *quantized_xb = new uint8_t[nb * sq.code_size];
  sq.sa_encode(nb, xb, quantized_xb);
  for (int i = 0; i < nb; i++) {
    hnsw.addPoint(quantized_xb + i * sq.code_size, i);
  }

  omp_set_num_threads(1);
  int k = 500;
  int efs = k;
  nlohmann::json json;
  {  // search xq
    auto initial_ncomp = hnsw.metric_distance_computations.load();
    auto initial_nhops = hnsw.metric_hops.load();

    uint8_t *quantized_xq = new uint8_t[nq * sq.code_size];
    sq.sa_encode(nq, xq, quantized_xq);

    std::vector<std::priority_queue<std::pair<int, hnswlib::labeltype>>> results(nq);

    auto search_beg = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nq; i++) {
      results[i] = hnsw.searchKnn(quantized_xq + i * sq.code_size, k);
    }
    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_beg).count();

    int tp = 0;
    for (int i = 0; i < nq; i++) {
      auto gt_min = dist_func(xq + i * d, xb + gt[i * nk] * d, &d);
      auto gt_max = dist_func(xq + i * d, xb + gt[i * nk + k - 1] * d, &d);
      while (results[i].size()) {
        auto pair = results[i].top();
        results[i].pop();
        auto j = pair.second;
        auto d = pair.first;
        if (std::find(gt + i * nk, gt + i * nk + k, j) != gt + i * nk + k) {
          tp++;
        }
      }
    }
    json[fmt::to_string(efs)]["recall"] = (double)tp / nq / k;
    json[fmt::to_string(efs)]["qps"] = nq * 1000000. / search_time;
    json[fmt::to_string(efs)]["num_computations"] =
        (double)(hnsw.metric_distance_computations.load() - initial_ncomp) / nq;
    json[fmt::to_string(efs)]["nhops"] = (double)(hnsw.metric_hops.load() - initial_nhops) / nq;
    json["M"] = M;
    json["k"] = k;
  }

  std::ofstream out(root / out_json);
  out << json.dump(4);
  out.close();

  return 0;
}
