#include <fmt/core.h>
#include <fmt/format.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include "config.h"
#include "faiss/MetricType.h"
#include "methods/Proclus.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  po::options_description configs;
  int k;
  float l, r;
  string datacard;
  int nlist;
  int dproclus;
  configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  configs.add_options()("l", po::value<decltype(l)>(&l)->required());
  configs.add_options()("r", po::value<decltype(r)>(&r)->required());
  configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist)->required());
  configs.add_options()("dproclus", po::value<decltype(dproclus)>(&dproclus)->required());
  configs.add_options()("datacard", po::value<decltype(datacard)>(&datacard)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[datacard];
  int d = c.dim;

  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  vector<vector<labeltype>> hybrid_topks;
  load_hybrid_query_gt(c, {l}, {r}, k, hybrid_topks);

  fs::path ckp_root(CKPS);
  std::string medoids_ckp = fmt::format("{}-{}.medoids", nlist, dproclus);
  std::string subspace_ckp = fmt::format("{}-{}.subspaces", nlist, dproclus);
  std::string ranking_ckp = fmt::format("{}-{}.ranking", nlist, dproclus);
  Proclus proclus(nlist, d);

  if (fs::exists(ckp_root / "Proclus" / c.name / subspace_ckp)) {
    proclus.read_subspaces((ckp_root / "Proclus" / c.name / subspace_ckp).string());
  } else {
    fmt::print("Subspace file does not exist. Exitting...\n");
    return -1;
  }

  if (fs::exists(ckp_root / "Proclus" / c.name / medoids_ckp)) {
    proclus.read_medoids((ckp_root / "Proclus" / c.name / medoids_ckp).string());
  } else {
    fmt::print("Medoids file does not exist. Exiting...\n");
    return -1;
  }

  auto b_ranked_clusters = new faiss::idx_t[c.n_base];
  if (fs::exists(ckp_root / "Proclus" / c.name / ranking_ckp)) {
    std::ifstream in(ckp_root / "Proclus" / c.name / ranking_ckp);
    in.read((char *)b_ranked_clusters, c.n_base * sizeof(faiss::idx_t));
  } else {
    fmt::print("Ranking file does not exist. Exitting...\n");
    return -1;
  }

  float* distances = new float[c.n_queries * nlist];
  auto q_ranked_clusters = new faiss::idx_t[c.n_queries * nlist];
  proclus.search_l1(c.n_queries, xq, q_ranked_clusters, nlist);

  int32_t *hist = new int32_t[nlist];
  memset(hist, 0, sizeof(int32_t) * nlist); // Corrected sizeof(int) to sizeof(int32_t)
  for (int i = 0; i < c.n_queries; i++) {
    for (int j = 0; j < k; j++) {
      for (int r = 0; r < nlist; r++) {
        // int lo = i * nlist, hi = (i+1) * nlist;
        if (b_ranked_clusters[hybrid_topks[i][j]] == q_ranked_clusters[i * nlist + r]) {
          hist[r]++;
          break;
        }
      }
    }
  }

  string hist_file = fmt::format("proclus_top_{}_in_cluster_hist_{}_{}_{}_{}_{}.bin", k, c.name, nlist, dproclus, l, r);
  fs::path stat_root(STATS);
  fs::path hist_path = stat_root / hist_file;
  std::ofstream out(hist_path.c_str());
  for (int i = 0; i < nlist; i++) {
    out.write((char *)(hist + i), sizeof(int32_t));
  }

  return 0;
}