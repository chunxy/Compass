#include <omp.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <set>

#include <faiss/IndexScalarQuantizer.h>
#include <fmt/format.h>
#include "config.h"
#include "faiss/IndexHNSW.h"
#include "hnswlib/hnswlib.h"
#include "utils/card.h"
#include "utils/reader.h"

using idx_t = faiss::idx_t;

int main() {
  auto func = hnswlib::L2Sqr;
  extern DataCard siftsmall_1_1000_top500_float32;
  unsigned long d = siftsmall_1_1000_top500_float32.dim;  // dimension
  int nb = siftsmall_1_1000_top500_float32.n_base;        // database size
  int nq = siftsmall_1_1000_top500_float32.n_queries;     // nb of queries
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

  faiss::IndexHNSWSQ index(d, faiss::ScalarQuantizer::QuantizerType::QT_8bit, 4);
  index.train(nb, xb);
  index.add(nb, xb);

  omp_set_num_threads(1);
  int k = 500;
  index.hnsw.efSearch = std::max(k, index.hnsw.efSearch);
  {  // search xq
    idx_t *I = new idx_t[k * nq];
    float *D = new float[k * nq];

    auto search_beg = std::chrono::high_resolution_clock::now();
    // Returned D are sorted according to the distance in the quantization space,
    // which does not necessarily reflect the distance in the original space.
    index.search(nq, xq, k, D, I);
    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_beg);

    std::vector<float> recall_at_ks(nq);
    for (int i = 0; i < nq; i++) {
      auto set_I = std::set<idx_t>(I + i * k, I + (i + 1) * k);
      unsigned count = 0;
      for (int j = 0; j < k; j++) {
        if (set_I.count(gt[i * nk + j]) != 0) count++;
      }

      recall_at_ks[i] = (float)count / k;
      printf("Recall: %4g%%, ", recall_at_ks[i] * 100);
      auto largest_gt_dist = func(xb + d * gt[i * nk + k - 1], xq + d * i, &d);
      auto largest_qt_dist = D[(i + 1) * k - 1];
      auto corresp_dist = func(xb + d * I[(i + 1) * k - 1], xq + d * i, &d);
      printf(
          "largest GT dist: %7g, largest QT dist: %7g, corresp. dist: %7g\n",
          largest_gt_dist,
          largest_qt_dist,
          corresp_dist
      );
    }
    auto sum_of_recalls =
        std::accumulate(recall_at_ks.begin(), recall_at_ks.end(), decltype(recall_at_ks)::value_type(0));
    fmt::print("QPS: {} ", nq * 1e6 / search_duration.count());
    printf("Average Recall: %4g%%\n", sum_of_recalls / nq * 100);

    delete[] I;
    delete[] D;
  }

  delete[] xb;
  delete[] xq;

  return 0;
}
