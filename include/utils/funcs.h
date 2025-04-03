#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "card.h"
#include "hnswlib.h"
#include "json.hpp"
#include "methods/Pod.h"

using hnswlib::labeltype;
using std::vector;

void load_hybrid_data(const DataCard &c, float *&xb, float *&xq, uint32_t *&gt, vector<vector<float>> &attrs);

void load_hybrid_query_gt(
    const DataCard &c,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<vector<labeltype>> &hybrid_topks
);

void load_filter_data(
    const DataCard &c,
    float *&xb,
    float *&xq,
    uint32_t *&gt,
    vector<int> &blabels,
    vector<int> &qlabels
);

void load_filter_query_gt(const DataCard &c, const int k, vector<vector<labeltype>> &hybrid_topks);

void stat_selectivity(const vector<float> &attrs, const int l_bound, const int u_bound, int &nsat);

void stat_selectivity(
    const vector<vector<float>> &attrs,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    int &nsat
);

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
);

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
    const std::string &out_json
);

void collate_acorn_stats(
    const long time_in_ms,
    const vector<float> &rec_at_ks,
    const vector<float> &pre_at_ks,
    const std::string &out_json
);

nlohmann::json collate_stat(
    const Stat &s,
    const int nb,
    const int nsat,
    const int k,
    const int nq,
    const int search_time,
    const int nthread
);

nlohmann::json collate_stat(
    const Stat &s,
    const int nb,
    const int nsat,
    const int k,
    const int nq,
    const int search_time,
    const int nthread,
    FILE* out
);