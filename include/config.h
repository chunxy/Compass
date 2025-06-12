#pragma once

#include <string>

// arithmetics
const float EPSILON = 1e-5;

// directories
const std::string WORKSPACE = "/home/chunxy/repos/Compass";
const std::string LOGS = WORKSPACE + "/logs_{}";
const std::string CKPS = WORKSPACE + "/checkpoints";
const std::string STATS = WORKSPACE + "/stats";

const std::string DATA = WORKSPACE + "/data";
const std::string RAW_DATA = DATA + "/raw";
const std::string ATTR_DATA = DATA + "/attr";
const std::string GT_DATA = DATA + "/gt";

// attribute paths
const std::string VALUE_PATH_TMPL = ATTR_DATA + "/{}_{:d}_{:d}.value.bin";    // {name}_{dim}_{range}
const std::string BLABEL_PATH_TMPL = ATTR_DATA + "/{}_base_{:d}.label.bin";   // {name}_base_{range}
const std::string QLABEL_PATH_TMPL = ATTR_DATA + "/{}_query_{:d}.label.bin";  // {name}_query_{range}
// groundtruth paths
const std::string HYBRID_GT_PATH_TMPL = GT_DATA + "/{}_{}_{}_{}_{}.hybrid.gt";  // {name}_{range}_{l}_{r}_{k}
const std::string FILTER_GT_PATH_TMPL = GT_DATA + "/{}_{}_{}.filter.gt";        // {name}_{range}_{k}
// workload names
const std::string HYBRID_WORKLOAD_TMPL = "{}_{}_{}_{}_{}";  // {method}_{range}_{l}_{r}_{k}
const std::string FILTER_WORKLOAD_TMPL = "{}_{}_{}";        // {method}_{range}_{k}
// index-related names
const std::string COMPASS_IVF_CHECKPOINT_TMPL = "{}.ivf";                   // {nlist}
const std::string COMPASS_IVF_IMI_CHECKPOINT_TMPL = "{}_{}.imi";            // {nsub}_{nbits}
const std::string COMPASS_IVF_IMI_RANK_CHECKPOINT_TMPL = "{}_{}_{}.rank";   // {nb}_{nsub}_{nbits}
const std::string COMPASS_RANK_CHECKPOINT_TMPL = "{}_{}.rank";              // {nb}_{nlist}
const std::string COMPASS_GRAPH_CHECKPOINT_TMPL = "{}_{}.hnsw";             // {M}_{efc}
const std::string COMPASS_CGRAPH_CHECKPOINT_TMPL = "{}_{}_{}.hnsw";         // {M}_{efc}_{nlist}
const std::string COMPASS_X_IVF_CHECKPOINT_TMPL = "{}-{}.x.ivf";            // {nlist}-{dx}
const std::string COMPASS_X_RANK_CHECKPOINT_TMPL = "{}-{}-{}.x.rank";       // {nb}-{nlist}-{dx}
const std::string COMPASS_X_CGRAPH_CHECKPOINT_TMPL = "{}-{}-{}-{}.x.hnsw";  // {M}-{efc}-{nlist}-{dx}
