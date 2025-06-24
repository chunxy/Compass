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
#include "methods/basis/ReentrantHNSW.h"
#include "utils/Pod.h"
#include "utils/card.h"
#include "utils/funcs.h"
#include "utils/reader.h"

namespace fs = boost::filesystem;
using namespace std::chrono;
using std::pair;
using std::priority_queue;
using std::set;
using std::vector;

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

  time_t ts = time(nullptr);
  auto tm = localtime(&ts);
  std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
  fs::path root("/home/chunxy/repos/Compass/scratches/test-reentrant-hnsw");

  fmt::print("Saving to {}.\n", (root / out_json).string());

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
  for (auto efs : {10, 20, 50, 100, 150, 200}) {
    comp->setEf(efs);
    int ncomp = 0;
    double recall = 0;
    long search_time = 0;
    vector<bool> is_graph_ppsl(nb);

    for (int j = 0; j < nq; j++) {
      auto search_start = high_resolution_clock::system_clock::now();

      priority_queue<pair<float, labeltype>> top_candidates;
      priority_queue<pair<float, labeltype>> candidate_set;
      vector<bool> visited(comp->cur_element_count, false);
      {
        tableint curr_obj = comp->enterpoint_node_;
        float curr_dist =
            comp->fstdistfunc_((float *)xq + j * d, comp->getDataByInternalId(curr_obj), comp->dist_func_param_);

        for (int level = comp->maxlevel_; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;

            data = (unsigned int *)comp->get_linklist(curr_obj, level);
            int size = comp->getListCount(data);

            tableint *datal = (tableint *)(data + 1);
            for (int i = 0; i < size; i++) {
              tableint cand = datal[i];

              if (cand < 0 || cand > comp->max_elements_) throw std::runtime_error("cand error");
              float dist = comp->fstdistfunc_(xq + j * d, comp->getDataByInternalId(cand), comp->dist_func_param_);

              if (dist < curr_dist) {
                curr_dist = dist;
                curr_obj = cand;
                changed = true;
              }
            }
          }
        }
        visited[curr_obj] = true;
        candidate_set.emplace(-curr_dist, curr_obj);
        top_candidates.emplace(curr_dist, curr_obj);
      }

      while (true) {
        comp->ReentrantSearchKnn(
            (float *)xq + j * d, k, -1, top_candidates, candidate_set, visited, nullptr, ncomp, is_graph_ppsl
        );
        if (top_candidates.size() >= efs)  // to revert
          break;
      }

      auto search_stop = high_resolution_clock::system_clock::now();
      search_time += duration_cast<microseconds>(search_stop - search_start).count();

      set<labeltype> rz_indices, gt_indices, rz_gt_interse;
      while (top_candidates.size() > k) {
        top_candidates.pop();
      }
      while (!top_candidates.empty()) {
        auto pair = top_candidates.top();
        top_candidates.pop();
        auto i = pair.second;
        auto d = pair.first;
        rz_indices.insert(i);
      }

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
  }

  std::ofstream ofs((root / out_json).c_str());
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}