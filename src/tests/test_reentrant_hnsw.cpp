#include <fmt/chrono.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <omp.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "json.hpp"
#include "methods/Pod.h"
#include "methods/ReentrantHNSW.h"
#include "utils/card.h"
#include "utils/funcs.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using std::priority_queue;
using std::vector;

auto dist_func = hnswlib::L2Sqr;

int main(int argc, char **argv) {
  // IvfGraph1dArgs args(argc, argv);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card["audio_1_10000_float32"];
  size_t d = c.dim;          // This has to be size_t due to dist_func() call.
  int nb = c.n_base;         // number of database vectors
  int nq = c.n_queries;      // number of queries
  int ng = c.n_groundtruth;  // number of computed groundtruth entries
  // assert(nq % batchsz == 0);
  int M = 32, efc = 200;
  int k = 100;

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path root("/home/chunxy/repos/Compass/src/tests/test_reentrant_hnsw");

  fmt::print("Saving to {}.\n", (root / out_json).string());

  // Load data.
  float *xb, *xq;
  uint32_t *gt;
  vector<vector<float>> _attrs;
  load_hybrid_data(c, xb, xq, gt, _attrs);
  fmt::print("Finished loading data.\n");

  // Load groundtruth for hybrid search.
  vector<vector<labeltype>> hybrid_topks(nq);
  load_hybrid_query_gt(c, {0}, vector<float>{10000}, k, hybrid_topks);
  fmt::print("Finished loading groundtruth.\n");

  L2Space l2space(d);
  ReentrantHNSW<float> *comp;

  string index_file = fmt::format("{}_M_{}_efc_{}.hnsw", c.name, M, efc);
  fs::path ckp_path = root / index_file;
  if (fs::exists(ckp_path)) {
    comp = new ReentrantHNSW<float>(&l2space, ckp_path.string(), false, nb);
  } else {
    comp = new ReentrantHNSW<float>(&l2space, nb, M, efc);
    auto build_index_start = high_resolution_clock::now();
    for (int i = 0; i < nb; i++) comp->addPoint(xb + i * d, i);
    auto build_index_stop = high_resolution_clock::now();
    comp->saveIndex(ckp_path.string());
  }
  fmt::print("Finished loading/building index\n");

  nlohmann::json json;
  for (auto efs : {100, 200}) {
    comp->setEf(efs);
    int ncomp = 0;
    vector<bool> is_graph_ppsl(nb);
    auto search_start = high_resolution_clock::system_clock::now();
    for (int j = 0; j < nq; j++) {
      priority_queue<pair<float, int64_t>> top_candidates;
      priority_queue<pair<float, int64_t>> candidate_set;

      vector<bool> visited(comp->cur_element_count, false);

      {
        tableint currObj = comp->enterpoint_node_;
        float curdist = comp->fstdistfunc_(
            (float *)xq + j * d, comp->getDataByInternalId(comp->enterpoint_node_), comp->dist_func_param_
        );

        for (int level = comp->maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)comp->get_linklist(currObj, level);
            int size = comp->getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > comp->max_elements_) throw std::runtime_error("cand error");
              float dist =
                  comp->fstdistfunc_(xq + j * d, comp->getDataByInternalId(cand), comp->dist_func_param_);

              if (dist < curdist) {
                curdist = dist;
                currObj = cand;
                changed = true;
              }
            }
          }
        }

        candidate_set.emplace(-curdist, currObj);
      }

      while (true) {
        comp->ReentrantSearchKnn(
            (float *)xq + j * d, k, -1, top_candidates, candidate_set, visited, NULL, ncomp, is_graph_ppsl
        );
        if (top_candidates.size() >= efs)  // to revert
          break;
      }
    }
    auto search_stop = high_resolution_clock::system_clock::now();
    auto search_time = duration_cast<microseconds>(search_stop - search_start).count();

    double recall = 0;
    for (int j = 0; j < nq; j++) {
      auto results = comp->searchKnn(xq + j * d, k);

      std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;
      int ivf_ppsl_in_rz = 0, graph_ppsl_in_rz = 0;
      while (results.size()) {
        auto pair = results.top();
        results.pop();
        auto i = pair.second;
        auto d = pair.first;
        assert(rz_indices.find(i) == rz_indices.end());
        rz_indices.insert(i);
      }

      int ivf_ppsl_in_tp = 0, graph_ppsl_in_tp = 0;
      for (auto i : hybrid_topks[j]) {
        gt_indices.insert(i);
      }
      auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
      auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);

      std::set_intersection(
          gt_indices.begin(),
          gt_indices.end(),
          rz_indices.begin(),
          rz_indices.end(),
          std::inserter(rz_gt_interse, rz_gt_interse.begin())
      );

      // stat.rec_at_ks[j] = (double)rz_gt_interse.size() / gt_indices.size();
      recall += (double)rz_gt_interse.size() / rz_indices.size();
    }
    json[fmt::to_string(efs)]["recall"] = recall / nq;
    json[fmt::to_string(efs)]["qps"] = nq * 1000000. / search_time;
  }

  std::ofstream ofs((root / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}