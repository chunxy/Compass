#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <boost/filesystem.hpp>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include "config.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  IvfGraph2dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[args.datacard];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  // assert(nq % args.batchsz == 0);

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  float *attrs;
  load_hybrid_data(c, xb, xq, gt, attrs);
  fmt::print("Finished loading data.\n");

  // Load query range and groundtruth for hybrid search.
  if (c.type == "skewed" || c.type == "correlated" || c.type == "anticorrelated" || c.type == "real") {
    args.l_bounds.resize(c.n_queries * c.attr_dim);
    args.u_bounds.resize(c.n_queries * c.attr_dim);
    std::string rg_path = fmt::format(HYBRID_RG_REVISION_PATH_TMPL, c.name, c.attr_dim, c.attr_range, c.type);
    auto rg = load_float32(rg_path, c.n_queries * 2, c.attr_dim);
    memcpy(args.l_bounds.data(), rg, c.n_queries * c.attr_dim * sizeof(float));
    memcpy(args.u_bounds.data(), rg + c.n_queries * c.attr_dim, c.n_queries * c.attr_dim * sizeof(float));
  } else if (c.type == "onesided" || c.type == "point" || c.type == "negation") {
    args.l_bounds.resize(c.n_queries);
    std::string rg_path = fmt::format(HYBRID_RG_REVISION_PATH_TMPL, c.name, c.attr_dim, c.attr_range, c.type);
    auto rg = load_float32(rg_path, c.n_queries, c.attr_dim);
    memcpy(args.l_bounds.data(), rg, c.n_queries * sizeof(float));
  }

  args.k = 100;  // We find top-100 for Navix.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt_revision(c, args.k, hybrid_topks);
  fmt::print("Finished loading query range and groundtruth.\n");

  std::stringstream query_file_ss, gt_file_ss;
  query_file_ss << "queries_" << c.type << ".txt";
  gt_file_ss << "gt_" << c.type << ".bin";

  fs::path data("/home/chunxy/repos/Compass/data/navix");
  fs::create_directories(data / c.name / "bench_data");
  std::string query_file = query_file_ss.str();
  std::string gt_file = gt_file_ss.str();
  std::ofstream query_ofs((data / c.name / "bench_data" / query_file).string());
  std::ofstream gt_ofs((data / c.name / "bench_data" / gt_file).string());
  fmt::print("Saving to {}\n", (data / c.name / "bench_data" / query_file).string());
  fmt::print("Saving to {}\n", (data / c.name / "bench_data" / gt_file).string());

  hnswlib::L2Space space(d);

  for (int i = 0; i < 200; i++) {
    float *query_vec = xq + i * d;

    std::stringstream predicate;
    if (c.type == "skewed") {
      predicate << "e.skewed >= " << args.l_bounds[i] << " AND e.skewed <= " << args.u_bounds[i];
    } else if (c.type == "correlated") {
      predicate << "e.correlated1 >= " << args.l_bounds[i * 2] << " AND e.correlated1 <= " << args.u_bounds[i * 2];
      predicate << " AND e.correlated2 >= " << args.l_bounds[i * 2 + 1]
                << " AND e.correlated2 <= " << args.u_bounds[i * 2 + 1];
    } else if (c.type == "anticorrelated") {
      predicate << "e.anticorrelated1 >= " << args.l_bounds[i * 2]
                << " AND e.anticorrelated1 <= " << args.u_bounds[i * 2];
      predicate << " AND e.anticorrelated2 >= " << args.l_bounds[i * 2 + 1]
                << " AND e.anticorrelated2 <= " << args.u_bounds[i * 2 + 1];
    } else if (c.type == "real") {
      predicate << std::fixed << "e.real1 >= " << args.l_bounds[i * 2] << " AND e.real1 <= " << args.u_bounds[i * 2];
      predicate << " AND e.real2 >= " << args.l_bounds[i * 2 + 1] << " AND e.real2 <= " << args.u_bounds[i * 2 + 1];
    } else if (c.type == "onesided") {
      predicate << "e.skewed >= " << args.l_bounds[i];
    } else if (c.type == "point") {
      predicate << "e.skewed = " << args.l_bounds[i];
    } else if (c.type == "negation") {
      predicate << "e.skewed <> " << args.l_bounds[i];
    } else {
      throw std::runtime_error("Unsupported type: " + c.type);
    }

    std::stringstream ss;
    ss << "[";
    for (int j = 0; j < d - 1; j++) {
      ss << std::fixed << std::setw(10) << std::setprecision(8) << query_vec[j] << ",";
    }
    ss << std::fixed << std::setprecision(8) << query_vec[d - 1] << "]";
    std::string vector_string = ss.str();

    // Replace the "-" with "_" in the c.name.
    std::string name = c.name;
    if (name.find("-") != std::string::npos) {
      name = name.replace(name.find("-"), 1, "_");
    }
    if (name != "flickr" && name != "deep10m") {
      name += fmt::format("_{}", 4);  // use the 4d database
    }
    std::string query = fmt::format(
        "MATCH (e:{}) WHERE {} "
        "CALL ANN_SEARCH(e.embedding, {}, <maxK>, <efsearch>, <useQ>, "
        "<knn>, <searchType>) RETURN e.id;\n",
        name,
        predicate.str(),
        vector_string
    );
    query_ofs << query;

    vector<labeltype> topk = hybrid_topks[i];
    for (int j = 0; j < topk.size(); j++) {
      int64_t id = topk[j];
      // Write 8-byte id to gt_ofs in binary.
      gt_ofs.write((char *)&id, sizeof(id));
    }
  }
  query_ofs.flush();
  gt_ofs.flush();
}