#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include "config.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "json.hpp"
#include "methods/basis/IterativeSearch.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/reader.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;
using namespace std::chrono;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card["sift_1_10000_float32"];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  int nlist = 10000;
  int M = 4, efc = 200;
  int k = 500;
  int batch_k = 50, initial_efs = 50;
  vector<int> delta_efs_s = {50, 100, 200};

  po::options_description optional_configs("Optional");
  optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
  optional_configs.add_options()("k", po::value<decltype(k)>(&k));
  optional_configs.add_options()("M", po::value<decltype(M)>(&M));
  optional_configs.add_options()("batch_k", po::value<decltype(batch_k)>(&batch_k));
  optional_configs.add_options()("initial_efs", po::value<decltype(initial_efs)>(&initial_efs));
  optional_configs.add_options()("delta_efs", po::value<decltype(delta_efs_s)>(&delta_efs_s)->multitoken());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, optional_configs), vm);
  po::notify(vm);

  auto xq = new float[nq * d];
  FVecItrReader reader(c.query_path);
  for (int i = 0; i < nq; i++) {
    auto next = reader.Next();
    memcpy(xq + i * d, next.data(), d * sizeof(float));
  }

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path log_root("/home/chunxy/repos/Compass/scratches/test-iterative-hnsw");
  fs::path ckp_root("/home/chunxy/repos/Compass/checkpoints/CompassR1d/" + c.name);
  fmt::print("Saving to {}.\n", (log_root / out_json).string());

  auto ivf_index = fmt::format(COMPASS_IVF_CHECKPOINT_TMPL, nlist);
  auto ivf_file = fopen((ckp_root / ivf_index).c_str(), "r");
  auto ivf = dynamic_cast<faiss::IndexIVFFlat *>(faiss::read_index(ivf_file));

  IterativeSearch<float> *comp;
  auto cgraph_index = fmt::format(COMPASS_CGRAPH_CHECKPOINT_TMPL, nlist, M, efc);
  fs::path cgraph_path = ckp_root / cgraph_index;
  if (!fs::exists(cgraph_path)) {
    throw std::runtime_error("Index file not found.");
  }
  comp = new IterativeSearch<float>(nb, d, cgraph_path.string(), new L2Space(d));
  fmt::print("Finished loading/building index\n");

  nlohmann::json json;
  for (auto efs : delta_efs_s) {
    comp->SetSearchParam(batch_k, efs);
    double recall = 0;
    double ncomp = 0;
    double search_time = 0;
    nq = 1000;
    for (int j = 0; j < nq; j++) {
      std::set<labeltype> graph_rz, ivf_rz, rz_interse;

      auto open_beg = high_resolution_clock::system_clock::now();
      IterativeSearchState<float> state = comp->Open(xq + j * d, k);
      auto open_end = high_resolution_clock::system_clock::now();
      search_time += duration_cast<microseconds>(open_end - open_beg).count();

      while (graph_rz.size() < k) {
        auto search_beg = high_resolution_clock::system_clock::now();
        auto pair = comp->Next(&state);
        auto search_end = high_resolution_clock::system_clock::now();
        search_time += duration_cast<microseconds>(search_end - search_beg).count();

        if (pair.first == -1 && pair.second == -1) {
          break;
        }
        auto i = pair.second;
        auto d = pair.first;
        graph_rz.insert(i);
      }

      ncomp += comp->GetNcomp(&state);
      comp->Close(&state);

      auto label = new faiss::idx_t[k];
      ivf->quantizer->assign(1, xq + j * d, label, k);
      for (int i = 0; i < k; i++) {
        ivf_rz.insert(label[i]);
      }
      std::set_intersection(
          ivf_rz.begin(), ivf_rz.end(), graph_rz.begin(), graph_rz.end(), std::inserter(rz_interse, rz_interse.begin())
      );

      recall += (double)rz_interse.size() / k;
    }
    json[fmt::to_string(efs)]["recall"] = recall / nq;
    json[fmt::to_string(efs)]["qps"] = nq * 1000000. / search_time;
    json[fmt::to_string(efs)]["ncomp"] = ncomp / nq;
    json["M"] = M;
    json["k"] = k;
    json["batch_k"] = batch_k;
    json["initial_efs"] = initial_efs;
  }

  std::ofstream ofs((log_root / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}