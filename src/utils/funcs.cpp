#include "utils/funcs.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>
#include "config.h"
#include "json.hpp"
#include "utils/card.h"
#include "utils/reader.h"

using std::vector;

void load_hybrid_data(
    const DataCard &c,
    float *&xb,
    float *&xq,
    uint32_t *&gt,
    vector<vector<float>> &attrs
) {
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
  // IVecItrReader groundtruth_it(c.groundtruth_path);
  // i = 0;
  // while (!groundtruth_it.HasEnded()) {
  //   auto next = groundtruth_it.Next();
  //   memcpy(gt + i * ng, next.data(), ng * sizeof(uint32_t));
  //   i++;
  // }

  std::string attr_path = fmt::format(VALUE_PATH_TMPL, c.name, c.attr_dim, c.attr_range);
  BinaryAttrReader<float> reader(attr_path);
  attrs = reader.GetAttrs();
}

void load_hybrid_query_gt(
    const DataCard &c,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<vector<labeltype>> &hybrid_topks
) {
  std::string gt_path = fmt::format(HYBRID_GT_PATH_TMPL, c.name, c.attr_range, l_bounds, u_bounds, 100);

  assert(std::ifstream(gt_path).good());
  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    assert(k <= topk.size());
    hybrid_topks[i].resize(k);
    for (int j = 0; j < k; j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }
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
  IVecItrReader groundtruth_it(c.groundtruth_path);
  i = 0;
  while (!groundtruth_it.HasEnded()) {
    auto next = groundtruth_it.Next();
    memcpy(gt + i * ng, next.data(), ng * sizeof(uint32_t));
    i++;
  }

  std::string blabel_path = fmt::format(BLABEL_PATH_TMPL, c.name, c.attr_range);
  BinaryAttrReader<int32_t> blabel_reader(blabel_path);
  auto _blabels = blabel_reader.GetAttrs();
  blabels.resize(_blabels.size());
  for (size_t i = 0; i < blabels.size(); i++) blabels[i] = _blabels[i][0];

  std::string qlabel_path = fmt::format(BLABEL_PATH_TMPL, c.name, c.attr_range);
  BinaryAttrReader<int32_t> qlabel_reader(qlabel_path);
  auto _qlabels = qlabel_reader.GetAttrs();
  qlabels.resize(_qlabels.size());
  for (size_t i = 0; i < qlabels.size(); i++) qlabels[i] = _qlabels[i][0];
}

void load_filter_query_gt(const DataCard &c, const int k, vector<vector<labeltype>> &hybrid_topks) {
  std::string gt_path = fmt::format(FILTER_GT_PATH_TMPL, c.name, c.attr_range, k);

  assert(std::ifstream(gt_path).good());
  hybrid_topks.resize(c.n_queries);
  int i = 0;
  IVecItrReader groundtruth_it(gt_path);
  while (!groundtruth_it.HasEnded()) {
    auto topk = groundtruth_it.Next();
    hybrid_topks[i].resize(topk.size());
    for (int j = 0; j < topk.size(); j++) {
      hybrid_topks[i][j] = topk[j];
    }
    i++;
  }
}

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

void collate_compass_stats(
    const long time_in_ms,
    const vector<float> &rec_at_ks,
    const vector<float> &pre_at_ks,
    const vector<float> &ivf_ppsl_qlty,
    const vector<float> &ivf_ppsl_rate,
    const vector<float> &perc_of_ivf_ppsl_in_tp,
    const vector<float> &perc_of_ivf_ppsl_in_rz,
    const vector<float> &linear_scan_rate,
    const vector<int> &ivf_ppsl_nums,
    const vector<long> &context_ts,
    const std::string &out_json
) {
  int nq = rec_at_ks.size();
  auto sum_of_rec = std::accumulate(rec_at_ks.begin(), rec_at_ks.end(), 0.);
  auto sum_of_pre = std::accumulate(pre_at_ks.begin(), pre_at_ks.end(), 0.);
  auto sum_of_num = std::accumulate(ivf_ppsl_nums.begin(), ivf_ppsl_nums.end(), 0.);
  auto sum_of_qlty = std::accumulate(ivf_ppsl_qlty.begin(), ivf_ppsl_qlty.end(), 0.);
  auto sum_of_rate = std::accumulate(ivf_ppsl_rate.begin(), ivf_ppsl_rate.end(), 0.);
  auto sum_of_percentage_in_tp =
      std::accumulate(perc_of_ivf_ppsl_in_tp.begin(), perc_of_ivf_ppsl_in_tp.end(), 0.);
  auto sum_of_percentage_in_rz =
      std::accumulate(perc_of_ivf_ppsl_in_rz.begin(), perc_of_ivf_ppsl_in_rz.end(), 0.);
  auto sum_of_linear_scan_rate = std::accumulate(linear_scan_rate.begin(), linear_scan_rate.end(), 0.);
  auto sum_of_context_switch_time = std::accumulate(context_ts.begin(), context_ts.end(), 0);
  fmt::print("Average Recall    : {:5.2f}%\n", sum_of_rec / nq * 100);
  fmt::print("Average Precision : {:5.2f}%\n", sum_of_pre / nq * 100);
  fmt::print("Average No. Ppsl  : {:5.2f}\n", sum_of_num / nq);
  fmt::print("Average Ppsl Rate : {:5.2f}%\n", sum_of_rate / nq * 100);
  fmt::print("Average Quality   : {:5.2f}%\n", sum_of_qlty / nq * 100);
  fmt::print("Average Perc in TP: {:5.2f}%\n", sum_of_percentage_in_tp / nq * 100);
  fmt::print("Average Perc in RZ: {:5.2f}%\n", sum_of_percentage_in_rz / nq * 100);
  fmt::print("Average Scan Rate : {:5.2f}%\n", sum_of_linear_scan_rate / nq * 100);

  nlohmann::json json;
  json["recall"] = rec_at_ks;
  json["precision"] = pre_at_ks;
  json["proposal_number"] = ivf_ppsl_nums;
  json["proposal_rate"] = ivf_ppsl_rate;
  json["proposal_quality"] = ivf_ppsl_qlty;
  json["percentage_in_tp"] = perc_of_ivf_ppsl_in_tp;
  json["percentage_in_rz"] = perc_of_ivf_ppsl_in_rz;
  json["linear_scan_rate"] = linear_scan_rate;

  json["aggregated"] = {
      {"recall", sum_of_rec / nq},
      {"precision", sum_of_pre / nq},
      {"proposal_number", sum_of_num / nq},
      {"proposal_rate", sum_of_rate / nq},
      {"proposal_quality", sum_of_qlty / nq},
      {"percentage_in_tp", sum_of_percentage_in_tp / nq},
      {"percentage_in_rz", sum_of_percentage_in_rz / nq},
      {"linear_scan_rate", sum_of_linear_scan_rate / nq},
      {"num_queries", nq},
      {"time_in_s", double(time_in_ms) / 1000},
      {"qps", double(nq) / time_in_ms * 1000},
      {"context_switch_in_s", (double)sum_of_context_switch_time / 1000}
  };
  std::ofstream ofs(out_json);
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}

void collate_acorn_stats(
    const long time_in_ms,
    const vector<float> &rec_at_ks,
    const vector<float> &pre_at_ks,
    const std::string &out_json
) {
  int nq = rec_at_ks.size();
  auto sum_of_rec = std::accumulate(rec_at_ks.begin(), rec_at_ks.end(), 0.);
  auto sum_of_pre = std::accumulate(pre_at_ks.begin(), pre_at_ks.end(), 0.);
  fmt::print("Average Recall    : {:5.2f}%\n", sum_of_rec / nq * 100);
  fmt::print("Average Precision : {:5.2f}%\n", sum_of_pre / nq * 100);

  nlohmann::json json;
  json["recall"] = rec_at_ks;
  json["precision"] = pre_at_ks;
  json["aggregated"] = {
      {"recall", sum_of_rec / nq},
      {"precision", sum_of_pre / nq},
      {"num_queries", nq},
      {"time_in_s", time_in_ms / 1000.},
      {"qps", (double)nq / time_in_ms * 1000},
  };
  std::ofstream ofs(out_json);
  ofs.write(json.dump(4).c_str(), json.dump(4).length());
}

nlohmann::json collate_stat(
    const Stat &s,
    const int nb,
    const int nsat,
    const int k,
    const int nq,
    const int search_time,
    const int nthread
) {
  for (int j = 0; j < s.rec_at_ks.size(); j++) {
    fmt::print("Query: {:d},\n", j);
    fmt::print("\tResult      : ");
    fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", s.rz_min_s[j], s.rz_max_s[j]);
    fmt::print("\tGround Truth: ");
    fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", s.gt_min_s[j], s.gt_max_s[j]);
    fmt::print("\tRecall: {:5.2f}%, ", s.rec_at_ks[j] * 100);
    fmt::print("Precision: {:5.2f}%, ", s.pre_at_ks[j] * 100);
    fmt::print("{:3d}/{:3d}/{:3d}\n", s.tp_s[j], s.rz_s[j], k);
    fmt::print("\tLatency in us         : {:d}\n", s.latencies[j]);
    fmt::print("\tNo. IVF Ppsl Rounds   : {:3d}\n", s.num_rounds[j]);
    fmt::print("\tNo. IVF Ppsl          : {:3d}\n", s.ivf_ppsl_nums[j]);
    fmt::print("\tNo. Graph Ppsl        : {:3d}\n", s.graph_ppsl_nums[j]);
    fmt::print("\tNo. Distance Comp     : {:3d}\n", s.num_computations[j]);
    fmt::print("\tIVF Proposal Rate     : {:3d}/{:3d}\n", s.ivf_ppsl_in_rz_s[j], s.ivf_ppsl_nums[j]);
    fmt::print("\tIVF Proposal Quality  : {:3d}/{:3d}\n", s.ivf_ppsl_in_tp_s[j], s.ivf_ppsl_nums[j]);
    fmt::print("\tGraph Proposal Rate   : {:3d}/{:3d}\n", s.graph_ppsl_in_rz_s[j], s.graph_ppsl_nums[j]);
    fmt::print("\tGraph Proposal Quality: {:3d}/{:3d}\n", s.graph_ppsl_in_tp_s[j], s.graph_ppsl_nums[j]);
    fmt::print("\tIVF Proposal in TP    : {:3d}/{:3d}\n", s.ivf_ppsl_in_tp_s[j], s.tp_s[j]);
    fmt::print("\tIVF Proposal in RZ    : {:3d}/{:3d}\n", s.ivf_ppsl_in_rz_s[j], s.rz_s[j]);
    fmt::print("\tLinear Scan Rate      : {:3d}/{:3d}\n", s.ivf_ppsl_nums[j], nsat);
  }
  auto sum_of_rec = std::accumulate(s.rec_at_ks.begin(), s.rec_at_ks.end(), 0.);
  auto sum_of_pre = std::accumulate(s.pre_at_ks.begin(), s.pre_at_ks.end(), 0.);
  auto sum_of_ivf_num = std::accumulate(s.ivf_ppsl_nums.begin(), s.ivf_ppsl_nums.end(), 0.);
  auto sum_of_ivf_qlty = std::accumulate(s.ivf_ppsl_qlty.begin(), s.ivf_ppsl_qlty.end(), 0.);
  auto sum_of_ivf_rate = std::accumulate(s.ivf_ppsl_rate.begin(), s.ivf_ppsl_rate.end(), 0.);
  auto sum_of_graph_num = std::accumulate(s.graph_ppsl_nums.begin(), s.graph_ppsl_nums.end(), 0.);
  auto sum_of_graph_qlty = std::accumulate(s.graph_ppsl_qlty.begin(), s.graph_ppsl_qlty.end(), 0.);
  auto sum_of_graph_rate = std::accumulate(s.graph_ppsl_rate.begin(), s.graph_ppsl_rate.end(), 0.);
  auto sum_of_perc_ivf_in_tp =
      std::accumulate(s.perc_of_ivf_ppsl_in_tp.begin(), s.perc_of_ivf_ppsl_in_tp.end(), 0.);
  auto sum_of_perc_ivf_in_rz =
      std::accumulate(s.perc_of_ivf_ppsl_in_rz.begin(), s.perc_of_ivf_ppsl_in_rz.end(), 0.);
  auto sum_of_linear_scan_rate = std::accumulate(s.linear_scan_rate.begin(), s.linear_scan_rate.end(), 0.);
  auto sum_of_latency = std::accumulate(s.latencies.begin(), s.latencies.end(), 0);
  auto sum_of_ctx_switch = std::accumulate(s.ctx_switch_time.begin(), s.ctx_switch_time.end(), 0);
  auto sum_of_num_comp = std::accumulate(s.num_computations.begin(), s.num_computations.end(), 0);
  auto sum_of_num_round = std::accumulate(s.num_rounds.begin(), s.num_rounds.end(), 0);

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
  json["latency"] = s.latencies;
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
      {"latency_in_s", (double)sum_of_latency / 1000000 / nq},
      {"context_switch_in_s", (double)sum_of_ctx_switch / 1000000 / nq},
      {"selectivity", (double)nsat / nb},
      {"time_in_s", (double)search_time / 1000000},
      {"qps", (double)nq / search_time * 1000000},
      {"num_threads", nthread},
      {"num_computations", (double)sum_of_num_comp / nq},
      {"num_rounds", (double)sum_of_num_round / nq}
  };
  return json;
}