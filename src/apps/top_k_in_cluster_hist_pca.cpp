#include <fmt/core.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include "config.h"
#include "faiss/IndexIVFFlat.h"
#include "faiss/IndexPreTransform.h"
#include "faiss/MetricType.h"
#include "faiss/index_io.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  po::options_description configs;
  int k;
  float l, r;
  string datacard;
  int nlist, dpca;
  configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  configs.add_options()("l", po::value<decltype(l)>(&l)->required());
  configs.add_options()("r", po::value<decltype(r)>(&r)->required());
  configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist)->required());
  configs.add_options()("dpca", po::value<decltype(dpca)>(&dpca)->required());
  configs.add_options()("datacard", po::value<decltype(datacard)>(&datacard)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[datacard];

  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  vector<vector<labeltype>> hybrid_topks;
  load_hybrid_query_gt(c, {l}, {r}, k, hybrid_topks);

  fs::path ckp_root(CKPS);
  string ivf_ckp = fmt::format(COMPASS_PCA_IVF_CHECKPOINT_TMPL, nlist, dpca);
  fs::path ivf_path = ckp_root / "PCA" / c.name / ivf_ckp;
  auto ivf_file = fopen(ivf_path.c_str(), "r");
  auto pca_ivf = dynamic_cast<faiss::IndexPreTransform *>(faiss::read_index(ivf_file));
  auto ivf = dynamic_cast<faiss::IndexIVFFlat *>(pca_ivf->index);

  auto q_ranked_clusters = new faiss::idx_t[c.n_queries * nlist];
  auto xqx = pca_ivf->apply_chain(c.n_queries, (float *)xq);
  ivf->quantizer->assign(c.n_queries, xqx, q_ranked_clusters, nlist);

  auto b_ranked_clusters = new faiss::idx_t[c.n_base];
  string rank_ckp = fmt::format(COMPASS_PCA_RANK_CHECKPOINT_TMPL, c.n_base, nlist, dpca);
  fs::path rank_path = ckp_root / "PCA" / c.name / rank_ckp;
  if (fs::exists(rank_path)) {
    auto rank_file = std::ifstream(rank_path.c_str());
    rank_file.read((char *)b_ranked_clusters, sizeof(faiss::idx_t) * c.n_base);
  }
  else {
    auto xbx = pca_ivf->apply_chain(c.n_base, (float *)xb);
    ivf->quantizer->assign(c.n_base, xbx, b_ranked_clusters, 1);
    // std::ofstream rank_file(rank_path.c_str());
    // rank_file.write((char *)b_ranked_clusters, sizeof(faiss::idx_t) * c.n_base);
  }

  int32_t *hist = new int32_t[nlist];
  memset(hist, 0, sizeof(int) * nlist);
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

  string hist_file = fmt::format("pca_top_{}_in_cluster_hist_{}_{}_{}_{}_{}.bin", k, c.name, nlist, dpca, l, r);
  fs::path stat_root(STATS);
  fs::path hist_path = stat_root / hist_file;
  std::ofstream out(hist_path.c_str());
  for (int i = 0; i < nlist; i++) {
    out.write((char *)(hist + i), sizeof(int32_t));
  }

  return 0;
}