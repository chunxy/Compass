#pragma once

#include <cstdint>
#include <queue>
#include <vector>
#include "card.h"
#include "hnswlib/hnswlib.h"
#include "json.hpp"
#include "utils/Pod.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

float *load_float32(const string &path, const int n, const int d);

uint32_t *load_uint32(const string &path, const int n, const int d);

void load_hybrid_data(const DataCard &c, float *&xb, float *&xq, uint32_t *&gt, vector<vector<float>> &attrs);

void load_hybrid_data(const DataCard &c, float *&xb, float *&xq, uint32_t *&gt, float *&attrs);

void load_hybrid_query_gt(
    const DataCard &c,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<vector<labeltype>> &hybrid_topks
);

void load_hybrid_query_gt_packed(
    const DataCard &c,
    const int perc,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    const int k,
    vector<int32_t> &l_ranges,
    vector<int32_t> &u_ranges,
    vector<vector<labeltype>> &hybrid_topks
);

void load_hybrid_query_gt_percents(
    const DataCard &c,
    const vector<int> &percents,
    const int k,
    vector<vector<labeltype>> &hybrid_topks,
    float *&l_bounds,
    float *&u_bounds
);

void load_hybrid_query_gt_revision(const DataCard &c, const int k, vector<vector<labeltype>> &hybrid_topks);

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

void stat_selectivity(
    const float *attrs,
    const int n,
    const int d,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    int &nsat
);

void stat_selectivity_revision(
    const float *attrs,
    const int n,
    const int d,
    const vector<float> &l_bounds,
    const vector<float> &u_bounds,
    int &nsat
);

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
);

nlohmann::json collate_stat(
    const Stat &s,
    const int nb,
    const int nsat,
    const int k,
    const int nq,
    const int search_time,
    const int nthread,
    FILE *out
);

void collate_acorn_stats(
    const long time_in_ms,
    const long ndis,
    const vector<float> &rec_at_ks,
    const vector<float> &pre_at_ks,
    const std::string &out_json,
    FILE *out
);