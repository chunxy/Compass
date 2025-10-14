#include "utils/funcs.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include "config.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/reader.h"

float *load_float32(const string &path, const int n, const int d) {
  std::ifstream in(path);
  if (!in.good()) {
    throw fmt::format("Failed to open file: {}", path);
  }
  auto storage = new float[n * d];
  in.read((char *)storage, sizeof(float) * n * d);
  return storage;
}

void load_hybrid_data(const DataCard &c, float *&xb, float *&xq, uint32_t *&gt, vector<vector<float>> &attrs) {
  auto d = c.dim, nb = c.n_base, nq = c.n_queries, ng = c.n_groundtruth;
  xb = new float[d * nb];
  xq = new float[d * nq];
  // gt = new uint32_t[ng * nq];
  int i;

  FVecItrReader base_it(c.base_path);
  i = 0;
  while (!base_it.HasEnded()) {
    auto next = base_it.Next();
    memcpy(xb + i * d, next.data(), d * sizeof(float));
    i++;
  }
  FVecItrReader query_it(c.query_path);
  i = 0;
  while (!query_it.HasEnded()) {
    auto next = query_it.Next();
    memcpy(xq + i * d, next.data(), d * sizeof(float));
    i++;
  }

  std::string attr_path = fmt::format(VALUE_PATH_TMPL, c.name, c.attr_dim, c.attr_range);
  AttrReaderToVector<float> reader(attr_path);
  attrs = reader.GetAttrs();
}

void load_hybrid_data(const DataCard &c, float *&xb, float *&xq, uint32_t *&gt, float *&attrs) {
  auto d = c.dim, nb = c.n_base, nq = c.n_queries, ng = c.n_groundtruth;
  xb = new float[d * nb];
  xq = new float[d * nq];
  // gt = new uint32_t[ng * nq];
  int i;

  FVecItrReader base_it(c.base_path);
  i = 0;
  while (!base_it.HasEnded()) {
    auto next = base_it.Next();
    memcpy(xb + i * d, next.data(), d * sizeof(float));
    i++;
  }
  FVecItrReader query_it(c.query_path);
  i = 0;
  while (!query_it.HasEnded()) {
    auto next = query_it.Next();
    memcpy(xq + i * d, next.data(), d * sizeof(float));
    i++;
  }

  std::string attr_path = fmt::format(VALUE_PATH_TMPL, c.name, c.attr_dim, c.attr_range);
  AttrReaderToRaw<float> reader(attr_path);
  attrs = reader.GetAttrs();
}

void load_hybrid_query_gt(
    const DataCard &c,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<vector<labeltype>> &hybrid_topks
) {
  std::string gt_path;
  if (l_bounds.size() == 1 || l_bounds.size() == 2) {  // Because the groundtruth was computed with k=100 for 1D and 2D.
    gt_path = fmt::format(HYBRID_GT_PATH_TMPL, c.name, c.attr_range, l_bounds, u_bounds, 100);
  } else {
    gt_path = fmt::format(HYBRID_GT_PATH_TMPL, c.name, c.attr_range, l_bounds, u_bounds, 100);
    std::ifstream gt_file(gt_path);
    if (!gt_file.good()) {
      gt_path = gt_path = fmt::format(HYBRID_GT_PATH_TMPL, c.name, c.attr_range, l_bounds, u_bounds, k);
    }
  }

  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    if (k > topk.size()) {
      throw fmt::format("k ({}) is greater than the size of the ground truth ({})", k, topk.size());
    }
    hybrid_topks[i].resize(k);
    for (int j = 0; j < k; j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }
}

void load_hybrid_query_gt_packed(
    const DataCard &c,
    const int perc,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<int32_t> &l_ranges,
    vector<int32_t> &u_ranges,
    vector<vector<labeltype>> &hybrid_topks
) {
  std::string gt_path;
  if (l_bounds.size() == 1 || l_bounds.size() == 2) {  // Because the groundtruth was computed with k=100 for 1D and 2D.
    gt_path = fmt::format(HYBRID_GT_CHEATING_PATH_TMPL, c.name, perc, l_bounds, u_bounds, 100);
  } else {
    gt_path = fmt::format(HYBRID_GT_CHEATING_PATH_TMPL, c.name, perc, l_bounds, u_bounds, 100);
    std::ifstream gt_file(gt_path);
    if (!gt_file.good()) {
      gt_path = gt_path = fmt::format(HYBRID_GT_CHEATING_PATH_TMPL, c.name, perc, l_bounds, u_bounds, k);
    }
  }

  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    if (k > topk.size()) {
      throw fmt::format("k ({}) is greater than the size of the ground truth ({})", k, topk.size());
    }
    hybrid_topks[i].resize(k);
    for (int j = 0; j < k; j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }

  l_ranges.resize(c.n_queries);
  u_ranges.resize(c.n_queries);
  std::string rg_path = fmt::format(HYBRID_RG_CHEATING_PATH_TMPL, c.name, perc, l_bounds, u_bounds, 100);
  std::ifstream rg_file(rg_path);
  if (!rg_file.good()) {
    throw fmt::format("Failed to open file: {}", rg_path);
  }
  rg_file.read((char *)l_ranges.data(), sizeof(int32_t) * c.n_queries);
  rg_file.read((char *)u_ranges.data(), sizeof(int32_t) * c.n_queries);
}

void load_hybrid_query_gt_percents(
    const DataCard &c,
    const vector<int> &percents,
    const int k,
    vector<vector<labeltype>> &hybrid_topks,
    float *&l_bounds,
    float *&u_bounds
) {
  std::string gt_path = fmt::format(HYBRID_GT_PATH_TMPL_NEO, c.name, c.attr_range, percents, 100);
  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    if (k > topk.size()) {
      throw fmt::format("k ({}) is greater than the size of the ground truth ({})", k, topk.size());
    }
    hybrid_topks[i].resize(k);
    for (int j = 0; j < k; j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }

  l_bounds = new float[c.attr_dim * c.n_queries];
  u_bounds = new float[c.attr_dim * c.n_queries];
  std::string rg_path = fmt::format(HYBRID_RG_PATH_TMPL, c.name, c.attr_range, percents);
  std::ifstream rg_file(rg_path);
  if (!rg_file.good()) {
    throw fmt::format("Failed to open file: {}", rg_path);
  }
  rg_file.read((char *)l_bounds, sizeof(float) * c.attr_dim * c.n_queries);
  rg_file.read((char *)u_bounds, sizeof(float) * c.attr_dim * c.n_queries);
}

void load_filter_data(
    const DataCard &c,
    float *&xb,
    float *&xq,
    uint32_t *&gt,
    vector<int> &blabels,
    vector<int> &qlabels
) {
  auto d = c.dim, nb = c.n_base, nq = c.n_queries, ng = c.n_groundtruth;
  xb = new float[d * nb];
  xq = new float[d * nq];
  gt = new uint32_t[ng * nq];
  int i;

  FVecItrReader base_it(c.base_path);
  i = 0;
  while (!base_it.HasEnded()) {
    auto next = base_it.Next();
    memcpy(xb + i * d, next.data(), d * sizeof(float));
    i++;
  }
  FVecItrReader query_it(c.query_path);
  i = 0;
  while (!query_it.HasEnded()) {
    auto next = query_it.Next();
    memcpy(xq + i * d, next.data(), d * sizeof(float));
    i++;
  }
  // IVecItrReader groundtruth_it(c.groundtruth_path);
  // i = 0;
  // while (!groundtruth_it.HasEnded()) {
  //   auto next = groundtruth_it.Next();
  //   memcpy(gt + i * ng, next.data(), ng * sizeof(uint32_t));
  //   i++;
  // }

  std::string blabel_path = fmt::format(BLABEL_PATH_TMPL, c.name, c.attr_range);
  AttrReaderToVector<int32_t> blabel_reader(blabel_path);
  auto _blabels = blabel_reader.GetAttrs();
  blabels.resize(_blabels.size());
  for (size_t i = 0; i < blabels.size(); i++) blabels[i] = _blabels[i][0];

  std::string qlabel_path = fmt::format(QLABEL_PATH_TMPL, c.name, c.attr_range);
  AttrReaderToVector<int32_t> qlabel_reader(qlabel_path);
  auto _qlabels = qlabel_reader.GetAttrs();
  qlabels.resize(_qlabels.size());
  for (size_t i = 0; i < qlabels.size(); i++) qlabels[i] = _qlabels[i][0];
}

void load_filter_query_gt(const DataCard &c, const int k, vector<vector<labeltype>> &hybrid_topks) {
  std::string gt_path = fmt::format(FILTER_GT_PATH_TMPL, c.name, c.attr_range, 100);
  std::ifstream gt_file(gt_path);
  if (!gt_file.good()) {
    gt_path = fmt::format(FILTER_GT_PATH_TMPL, c.name, c.attr_range, k);
  }

  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    if (k > topk.size()) {
      throw fmt::format("k ({}) is greater than the size of the ground truth ({})", k, topk.size());
    }
    hybrid_topks[i].resize(k);
    for (int j = 0; j < k; j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }
}

// 1D attribute
void stat_selectivity(const vector<float> &attrs, const int l_bound, const int u_bound, int &nsat) {
  nsat = 0;
  for (int i = 0; i < attrs.size(); i++) {
    if (l_bound <= attrs[i] && attrs[i] <= u_bound) nsat++;
  }
}

void stat_selectivity(
    const vector<vector<float>> &attrs,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    int &nsat
) {
  nsat = attrs.size();
  for (int i = 0; i < attrs.size(); i++) {
    for (int j = 0; j < l_bounds.size(); j++) {
      if (l_bounds[j] > attrs[i][j] || attrs[i][j] > u_bounds[j]) {
        nsat--;
        break;
      }
    }
  }
}

void stat_selectivity(
    const float *attrs,
    const int n,
    const int d,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    int &nsat
) {
  nsat = n;
  if (l_bounds.size() != d || u_bounds.size() != d) {
    throw std::invalid_argument("l_bounds and u_bounds must have the same dimension as d");
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (l_bounds[j] > attrs[i * d + j] || attrs[i * d + j] > u_bounds[j]) {
        nsat--;
        break;
      }
    }
  }
}

void collect_batch_metric(
    const vector<priority_queue<pair<float, labeltype>>> &results,  // indexed with i
    const BatchMetric &bm,                                          // indexed with i
    const vector<vector<labeltype>> &hybrid_topks,                  // indexed from curr
    const int curr,
    const vector<float> &gt_min_s,  // indexed with i
    const vector<float> &gt_max_s,  // indexed with i
    const float EPSILON,
    const int nsat,
    Stat &stat  // indexed from curr
) {
  for (int i = 0, j = curr; i < results.size(); i++, j++) {
    auto rz = results[i];
    std::set<int> rz_ids;
    auto metric = bm.qmetrics[i];
    float rz_min = std::numeric_limits<float>::max(), rz_max = std::numeric_limits<float>::min();
    int ivf_ppsl_in_rz = 0, graph_ppsl_in_rz = 0;
    int ivf_ppsl_in_tp = 0, graph_ppsl_in_tp = 0;
    int tp = 0;
    while (!rz.empty()) {
      auto pair = rz.top();
      rz.pop();
      auto id = pair.second;
      auto dist = pair.first;
      if (rz_ids.find(id) != rz_ids.end()) {
        fmt::print("Duplicate ID: {}\n", id);
        continue;
      }
      rz_ids.insert(id);
      if (metric.is_ivf_ppsl[id]) {
        ivf_ppsl_in_rz++;
      } else if (metric.is_graph_ppsl[id]) {
        graph_ppsl_in_rz++;
      }
      if (std::find(hybrid_topks[j].begin(), hybrid_topks[j].end(), id) != hybrid_topks[j].end() ||
          dist <= gt_max_s[i] + EPSILON) {
        if (metric.is_ivf_ppsl[id]) {
          ivf_ppsl_in_tp++;
        } else if (metric.is_graph_ppsl[id]) {
          graph_ppsl_in_tp++;
        }
        tp++;
      }
      rz_min = std::min(rz_min, dist);
      rz_max = std::max(rz_max, dist);
    }

    stat.ivf_ppsl_in_rz_s[j] = ivf_ppsl_in_rz;
    stat.graph_ppsl_in_rz_s[j] = graph_ppsl_in_rz;
    stat.rz_min_s[j] = rz_min;
    stat.rz_max_s[j] = rz_max;
    stat.gt_min_s[j] = gt_min_s[i];
    stat.gt_max_s[j] = gt_max_s[i];
    stat.ivf_ppsl_in_tp_s[j] = ivf_ppsl_in_tp;
    stat.graph_ppsl_in_tp_s[j] = graph_ppsl_in_tp;

    stat.tp_s[j] = tp;
    stat.rz_s[j] = rz.size();
    stat.rec_at_ks[j] = (double)stat.tp_s[j] / hybrid_topks[j].size();
    stat.pre_at_ks[j] = (double)stat.tp_s[j] / rz.size();

    stat.ivf_ppsl_nums[j] = std::accumulate(metric.is_ivf_ppsl.begin(), metric.is_ivf_ppsl.end(), 0);
    stat.graph_ppsl_nums[j] = std::accumulate(metric.is_graph_ppsl.begin(), metric.is_graph_ppsl.end(), 0);
    stat.ivf_ppsl_qlty[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_tp / stat.ivf_ppsl_nums[j] : 0;
    stat.ivf_ppsl_rate[j] = stat.ivf_ppsl_nums[j] != 0 ? (double)ivf_ppsl_in_rz / stat.ivf_ppsl_nums[j] : 0;
    stat.graph_ppsl_qlty[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_tp / stat.graph_ppsl_nums[j] : 0;
    stat.graph_ppsl_rate[j] = stat.graph_ppsl_nums[j] != 0 ? (double)graph_ppsl_in_rz / stat.graph_ppsl_nums[j] : 0;
    stat.perc_of_ivf_ppsl_in_tp[j] = stat.tp_s[j] != 0 ? (double)ivf_ppsl_in_tp / stat.tp_s[j] : 0;
    stat.perc_of_ivf_ppsl_in_rz[j] = (double)ivf_ppsl_in_rz / rz.size();
    stat.linear_scan_rate[j] = (double)stat.ivf_ppsl_nums[j] / nsat;
    stat.num_computations[j] = metric.ncomp;
    stat.num_computations_graph[j] = metric.ncomp_graph;
    stat.cg_num_computations[j] = metric.ncomp_cg;
    stat.num_rounds[j] = metric.nround;
    stat.num_clusters[j] = metric.ncluster;
    stat.num_recycled[j] = metric.nrecycled;
    stat.latencies[j] = metric.latency;
    stat.ivf_latencies[j] = metric.ivf_latency;
    stat.cg_latencies[j] = metric.cg_latency;
    stat.filter_latencies[j] = metric.filter_latency;
    stat.graph_latencies[j] = metric.graph_latency;
    stat.twohop_latencies[j] = metric.twohop_latency;
    stat.misc_latencies[j] = metric.ihnsw_latency;
    stat.comp_latencies[j] = metric.comp_latency;
  }
  stat.batch_time.push_back(bm.time);
  stat.batch_overhead.push_back(bm.overhead);
  stat.batch_cluster_search_time.push_back(bm.cluster_search_time);
}

nlohmann::json collate_stat(
    const Stat &s,
    const int nb,
    const int nsat,
    const int k,
    const int nq,
    const int search_time,  // in microseconds
    const int nthread,
    FILE *out
) {
  int nbatch = s.batch_time.size();
  if (nbatch == 0) nbatch = nq;
  int batch_sz = nq / nbatch;

  for (int j = 0; j < s.rec_at_ks.size(); j++) {
    fmt::print(out, "Query: {:d},\n", j);
    fmt::print(out, "\tResult      : ");
    fmt::print(out, "Min: {:9.2f}, Max: {:9.2f}\n", s.rz_min_s[j], s.rz_max_s[j]);
    fmt::print(out, "\tGround Truth: ");
    fmt::print(out, "Min: {:9.2f}, Max: {:9.2f}\n", s.gt_min_s[j], s.gt_max_s[j]);
    fmt::print(out, "\tRecall: {:5.2f}%, ", s.rec_at_ks[j] * 100);
    fmt::print(out, "Precision: {:5.2f}%, ", s.pre_at_ks[j] * 100);
    fmt::print(out, "{:3d}/{:3d}/{:3d}\n", s.tp_s[j], s.rz_s[j], k);
    fmt::print(out, "\tLatency in ns         : {:d}\n", s.latencies[j]);
    fmt::print(out, "\tBatch overhead in ns  : {:d}\n", s.batch_overhead[j / batch_sz] / batch_sz);
    fmt::print(out, "\tCG Latency in ns      : {:d}\n", s.cg_latencies[j]);
    fmt::print(out, "\tIVF Latency in ns     : {:d}\n", s.ivf_latencies[j]);
    fmt::print(out, "\tFilter Latency in ns  : {:d}\n", s.filter_latencies[j]);
    fmt::print(out, "\tGraph Latency in ns   : {:d}\n", s.graph_latencies[j]);
    fmt::print(out, "\tMisc Latency in ns    : {:d}\n", s.misc_latencies[j]);
    fmt::print(out, "\tTwohop Comp Lat. in ns: {:d}\n", s.comp_latencies[j]);
    fmt::print(out, "\tNo. IVF Ppsl Rounds   : {:3d}\n", s.num_rounds[j]);
    fmt::print(out, "\tNo. IVF Ppsl          : {:3d}\n", s.ivf_ppsl_nums[j]);
    fmt::print(out, "\tNo. Graph Ppsl        : {:3d}\n", s.graph_ppsl_nums[j]);
    fmt::print(out, "\tNo. Distance Comp     : {:3d}\n", s.num_computations[j]);
    fmt::print(out, "\tNo. Distance Comp Graph: {:3d}\n", s.num_computations_graph[j]);
    fmt::print(out, "\tIVF Proposal Rate     : {:3d}/{:3d}\n", s.ivf_ppsl_in_rz_s[j], s.ivf_ppsl_nums[j]);
    fmt::print(out, "\tIVF Proposal Quality  : {:3d}/{:3d}\n", s.ivf_ppsl_in_tp_s[j], s.ivf_ppsl_nums[j]);
    fmt::print(out, "\tGraph Proposal Rate   : {:3d}/{:3d}\n", s.graph_ppsl_in_rz_s[j], s.graph_ppsl_nums[j]);
    fmt::print(out, "\tGraph Proposal Quality: {:3d}/{:3d}\n", s.graph_ppsl_in_tp_s[j], s.graph_ppsl_nums[j]);
    fmt::print(out, "\tIVF Proposal in TP    : {:3d}/{:3d}\n", s.ivf_ppsl_in_tp_s[j], s.tp_s[j]);
    fmt::print(out, "\tIVF Proposal in RZ    : {:3d}/{:3d}\n", s.ivf_ppsl_in_rz_s[j], s.rz_s[j]);
    fmt::print(out, "\tLinear Scan Rate      : {:3d}/{:3d}\n", s.ivf_ppsl_nums[j], nsat);
  }
  auto sum_of_rec = std::accumulate(s.rec_at_ks.begin(), s.rec_at_ks.end(), 0.);
  auto sum_of_pre = std::accumulate(s.pre_at_ks.begin(), s.pre_at_ks.end(), 0.);
  auto sum_of_ivf_num = std::accumulate(s.ivf_ppsl_nums.begin(), s.ivf_ppsl_nums.end(), 0.);
  auto sum_of_ivf_qlty = std::accumulate(s.ivf_ppsl_qlty.begin(), s.ivf_ppsl_qlty.end(), 0.);
  auto sum_of_ivf_rate = std::accumulate(s.ivf_ppsl_rate.begin(), s.ivf_ppsl_rate.end(), 0.);
  auto sum_of_graph_num = std::accumulate(s.graph_ppsl_nums.begin(), s.graph_ppsl_nums.end(), 0.);
  auto sum_of_graph_qlty = std::accumulate(s.graph_ppsl_qlty.begin(), s.graph_ppsl_qlty.end(), 0.);
  auto sum_of_graph_rate = std::accumulate(s.graph_ppsl_rate.begin(), s.graph_ppsl_rate.end(), 0.);
  auto sum_of_perc_ivf_in_tp = std::accumulate(s.perc_of_ivf_ppsl_in_tp.begin(), s.perc_of_ivf_ppsl_in_tp.end(), 0.);
  auto sum_of_perc_ivf_in_rz = std::accumulate(s.perc_of_ivf_ppsl_in_rz.begin(), s.perc_of_ivf_ppsl_in_rz.end(), 0.);
  auto sum_of_linear_scan_rate = std::accumulate(s.linear_scan_rate.begin(), s.linear_scan_rate.end(), 0.);
  auto sum_of_num_cluster = std::accumulate(s.num_clusters.begin(), s.num_clusters.end(), 0l);
  auto sum_of_num_comp = std::accumulate(s.num_computations.begin(), s.num_computations.end(), 0l);
  auto sum_of_num_comp_graph = std::accumulate(s.num_computations_graph.begin(), s.num_computations_graph.end(), 0l);
  auto sum_of_num_round = std::accumulate(s.num_rounds.begin(), s.num_rounds.end(), 0l);
  auto sum_of_num_recycled = std::accumulate(s.num_recycled.begin(), s.num_recycled.end(), 0l);

  nlohmann::json json;
  json["recall"] = s.rec_at_ks;
  json["precision"] = s.pre_at_ks;
  json["ivf_proposal_number"] = s.ivf_ppsl_nums;
  json["ivf_proposal_rate"] = s.ivf_ppsl_rate;
  json["ivf_proposal_quality"] = s.ivf_ppsl_qlty;
  json["graph_proposal_number"] = s.graph_ppsl_nums;
  json["graph_proposal_rate"] = s.graph_ppsl_rate;
  json["graph_proposal_quality"] = s.graph_ppsl_qlty;
  json["perc_ivf_in_tp"] = s.perc_of_ivf_ppsl_in_tp;
  json["perc_ivf_in_rz"] = s.perc_of_ivf_ppsl_in_rz;
  json["linear_scan_rate"] = s.linear_scan_rate;

  json["batched"] = {
      {"batch_time_in_us", s.batch_time},
      {"batch_overhead_in_ns", s.batch_overhead},
      {"batch_cluster_search_time_in_ns", s.batch_cluster_search_time},
  };
  auto sum_of_latency = std::accumulate(s.latencies.begin(), s.latencies.end(), 0ll);
  auto sum_of_cg_latency = std::accumulate(s.cg_latencies.begin(), s.cg_latencies.end(), 0ll);
  auto sum_of_filter_latency = std::accumulate(s.filter_latencies.begin(), s.filter_latencies.end(), 0ll);
  auto sum_of_ivf_latency = std::accumulate(s.ivf_latencies.begin(), s.ivf_latencies.end(), 0ll);
  auto sum_of_graph_latency = std::accumulate(s.graph_latencies.begin(), s.graph_latencies.end(), 0ll);
  auto sum_of_twohop_latency = std::accumulate(s.twohop_latencies.begin(), s.twohop_latencies.end(), 0ll);
  auto sum_of_misc_latency = std::accumulate(s.misc_latencies.begin(), s.misc_latencies.end(), 0ll);
  auto sum_of_comp_latency = std::accumulate(s.comp_latencies.begin(), s.comp_latencies.end(), 0ll);
  auto sum_of_cg_ncomp = std::accumulate(s.cg_num_computations.begin(), s.cg_num_computations.end(), 0ll);

  auto sum_of_cluster_search_time =
      std::accumulate(s.batch_cluster_search_time.begin(), s.batch_cluster_search_time.end(), 0ll);

  json["aggregated"] = {
      {"recall", sum_of_rec / nq},
      {"precision", sum_of_pre / nq},
      {"ivf_proposal_number", sum_of_ivf_num / nq},
      {"ivf_proposal_rate", sum_of_ivf_rate / nq},
      {"ivf_proposal_quality", sum_of_ivf_qlty / nq},
      {"graph_proposal_number", sum_of_graph_num / nq},
      {"graph_proposal_rate", sum_of_graph_rate / nq},
      {"graph_proposal_quality", sum_of_graph_qlty / nq},
      {"perc_ivf_in_tp", sum_of_perc_ivf_in_tp / nq},
      {"perc_ivf_in_rz", sum_of_perc_ivf_in_rz / nq},
      {"linear_scan_rate", sum_of_linear_scan_rate / nq},
      {"num_queries", nq},
      {"selectivity", (double)nsat / nb},
      {"time_in_s", (double)search_time / 1000000},
      {"qps", (double)nq / search_time * 1000000},
      {"tampered_qps", (double)nq / (search_time - sum_of_cluster_search_time / 1000) * 1000000},
      {"num_threads", nthread},
      {"num_computations", (double)sum_of_num_comp / nq},
      {"num_computations_graph", (double)sum_of_num_comp_graph / nq},
      {"cg_num_computations", (double)sum_of_cg_ncomp / nq},
      {"num_clusters", (double)sum_of_num_cluster / nq},
      {"num_rounds", (double)sum_of_num_round / nq},
      {"num_recycled", (double)sum_of_num_recycled / nq},
      {"batch_sz", batch_sz},
      {"latency_in_ns", (double)sum_of_latency / nq},
      {"latency_ivf_in_ns", (double)sum_of_ivf_latency / nq},
      {"latency_cg_in_ns", (double)sum_of_cg_latency / nq},
      {"latency_filter_in_ns", (double)sum_of_filter_latency / nq},
      {"latency_graph_in_ns", (double)sum_of_graph_latency / nq},
      {"latency_twohop_in_ns", (double)sum_of_twohop_latency / nq},
      {"latency_misc_in_ns", (double)sum_of_misc_latency / nq},
      {"latency_twohop_comp_in_ns", (double)sum_of_comp_latency / nq},
  };
  return json;
}

void collate_acorn_stats(
    const long time_in_ms,
    const long ndis,
    const vector<float> &rec_at_ks,
    const vector<float> &pre_at_ks,
    const std::string &out_json,
    FILE *out
) {
  int nq = rec_at_ks.size();
  auto sum_of_rec = std::accumulate(rec_at_ks.begin(), rec_at_ks.end(), 0.);
  auto sum_of_pre = std::accumulate(pre_at_ks.begin(), pre_at_ks.end(), 0.);
  fmt::print(out, "Average Recall    : {:5.2f}%\n", sum_of_rec / nq * 100);
  fmt::print(out, "Average Precision : {:5.2f}%\n", sum_of_pre / nq * 100);

  nlohmann::json json;
  json["recall"] = rec_at_ks;
  json["precision"] = pre_at_ks;
  json["aggregated"] = {
      {"recall", sum_of_rec / nq},
      {"precision", sum_of_pre / nq},
      {"num_computations", double(ndis) / nq},
      {"num_queries", nq},
      {"time_in_s", time_in_ms / 1000.},
      {"qps", (double)nq / time_in_ms * 1000},
  };
  std::ofstream ofs(out_json);
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}