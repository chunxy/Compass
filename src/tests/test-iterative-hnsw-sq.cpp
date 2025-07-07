#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>

#include "faiss/IndexScalarQuantizer.h"
#include "json.hpp"
#include "methods/basis/IterativeSearch.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/reader.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;
using namespace std::chrono;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card["siftsmall_1_1000_top500_float32"];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  int M = 4, efc = 200;
  int k = 500;
  int batch_k = 10;
  vector<int> delta_efs_s = {100, 200};

  po::options_description optional_configs("Optional");
  optional_configs.add_options()("k", po::value<decltype(k)>(&k));
  optional_configs.add_options()("M", po::value<decltype(M)>(&M));
  optional_configs.add_options()("batch_k", po::value<decltype(batch_k)>(&batch_k));
  optional_configs.add_options()("delta_efs", po::value<decltype(delta_efs_s)>(&delta_efs_s)->multitoken());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, optional_configs), vm);
  po::notify(vm);

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path log_root("/home/chunxy/repos/Compass/scratches/test-iterative-hnsw-sq");
  fs::path ckp_root("/home/chunxy/repos/Compass/scratches/test-iterative-hnsw-sq");
  fmt::print("Saving to {}.\n", (log_root / out_json).string());

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<uint32_t>> hybrid_topks(nq);
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

  faiss::IndexScalarQuantizer sq(d, faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform, faiss::METRIC_L2);
  sq.train(nb, xb);
  sq.add(nb, xb);
  uint8_t *quantized_xb = new uint8_t[nb * sq.code_size];
  sq.sa_encode(nb, xb, quantized_xb);
  uint8_t *quantized_xq = new uint8_t[nq * sq.code_size];
  sq.sa_encode(nq, xq, quantized_xq);

  IterativeSearch<int> *comp;

  string index_file = fmt::format("{}_M_{}_efc_{}.{}bits.hnsw", c.name, M, efc, sq.code_size);
  fs::path ckp_path = ckp_root / index_file;
  if (fs::exists(ckp_path)) {
    comp = new IterativeSearch<int>(nb, d, ckp_path.string(), new L2SpaceB(d));
  } else {
    comp = new IterativeSearch<int>(nb, d, new L2SpaceB(d), M);
    for (int i = 0; i < nb; i++) {
      comp->hnsw_->addPoint(quantized_xb + i * sq.code_size, i);
    }
    comp->hnsw_->saveIndex(ckp_path.string());
  }
  fmt::print("Finished loading/building index\n");

  nlohmann::json json;
  for (auto efs : delta_efs_s) {
    comp->SetSearchParam(batch_k, efs);

    double recall = 0;
    double ncomp = 0;
    double search_time = 0;
    for (int j = 0; j < nq; j++) {
      std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;

      IterativeSearchState<int> *state = comp->Open((quantized_xq + j * sq.code_size), k);
      while (rz_indices.size() < k) {
        auto search_beg = high_resolution_clock::system_clock::now();
        auto pair = comp->Next(state);
        auto search_end = high_resolution_clock::system_clock::now();
        search_time += duration_cast<microseconds>(search_end - search_beg).count();

        if (pair.first == -1 && pair.second == -1) {
          break;
        }
        auto i = pair.second;
        auto d = pair.first;
        rz_indices.insert(i);
      }

      ncomp += comp->GetNcomp(state);
      comp->Close(state);

      for (int i = 0; i < k; i++) {
        gt_indices.insert(hybrid_topks[j][i]);
      }
      std::set_intersection(
          gt_indices.begin(),
          gt_indices.end(),
          rz_indices.begin(),
          rz_indices.end(),
          std::inserter(rz_gt_interse, rz_gt_interse.begin())
      );

      recall += (double)rz_gt_interse.size() / k;
    }
    json[fmt::to_string(efs)]["recall"] = recall / nq;
    json[fmt::to_string(efs)]["qps"] = nq * 1000000. / search_time;
    json[fmt::to_string(efs)]["ncomp"] = ncomp / nq;
    json["M"] = M;
    json["k"] = k;
    json["batch_k"] = batch_k;
  }

  std::ofstream ofs((log_root / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}