// #include <fmt/chrono.h>
// #include <fmt/core.h>
// #include <fmt/format.h>
// #include <boost/filesystem.hpp>
// #include <boost/program_options.hpp>
// #include <cassert>
// #include <chrono>
// #include <cstdint>
// #include <cstdio>
// #include <map>
// #include <set>
// #include <string>
// #include <utility>
// #include <vector>
// #include "AnnService/inc/Core/Common/BKTree.h"
// #include "config.h"
// #include "hnswlib.h"
// #include "methods/IndexIVFFlatBalanced.h"
// #include "utils/card.h"
// #include "utils/funcs.h"

// namespace po = boost::program_options;
// namespace fs = boost::filesystem;
// using namespace std::chrono;
// using std::vector;

// auto dist_func = hnswlib::L2Sqr;

// // int main() {
// //   SPTAG::COMMON::BKTree bktree;
// //   return 0;
// // }

// int main(int argc, char **argv) {
//   float l_bound, u_bound;
//   int k;
//   std::string dataname;
//   int nlist = 100;  // the number of coarse clusters
//   int nprobe = 100;

//   po::options_description configs;
//   po::options_description required_configs("Required");
//   // dataset parameter
//   required_configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
//   // search parameters
//   required_configs.add_options()("l", po::value<decltype(l_bound)>(&l_bound)->required());
//   required_configs.add_options()("r", po::value<decltype(u_bound)>(&u_bound)->required());
//   required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
//   po::options_description optional_configs("Optional");
//   // algorithm hyper-parameters
//   optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
//   optional_configs.add_options()("nprobe", po::value<decltype(nprobe)>(&nprobe));
//   // Merge required and optional configs.
//   configs.add(required_configs).add(optional_configs);
//   // Parse arguments.
//   po::variables_map vm;
//   po::store(po::parse_command_line(argc, argv, configs), vm);
//   po::notify(vm);

//   extern std::map<std::string, DataCard> name_to_card;
//   DataCard c = name_to_card[dataname];
//   size_t d = c.dim;          // This has to be size_t due to dist_func() call.
//   int nb = c.n_base;         // number of database vectors
//   int nq = c.n_queries;      // number of queries
//   int ng = c.n_groundtruth;  // number of computed groundtruth entries

//   time_t ts = time(nullptr);
//   auto tm = localtime(&ts);
//   std::string method = "compass_ivf";
//   std::string workload = fmt::format(HYBRID_WORKLOAD_TMPL, c.name, c.attr_range, l_bound, u_bound, k);
//   std::string param = fmt::format("nlist_{}_nprobe_{}", nlist, nprobe);
//   std::string out_text = fmt::format("{:%Y-%m-%d-%H-%M-%S}.log", *tm);
//   std::string out_json = fmt::format("{:%Y-%m-%d-%H-%M-%S}.json", *tm);
//   fs::path dir(LOGS);
//   fs::path log_dir = dir / method / workload / param;
//   fs::create_directories(log_dir);

//   fmt::print("Writing to {}.\n", (log_dir / out_text).string());
//   fmt::print("Saving to {}.\n", (log_dir / out_json).string());
//   // #ifndef NDEBUG
//   freopen((log_dir / out_text).c_str(), "w", stdout);
//   // #endif

//   // Load data.
//   float *xb, *xq;
//   uint32_t *gt;
//   vector<float> attrs;
//   load_hybrid_data(c, xb, xq, gt, attrs);
//   fmt::print("Finished loading data.\n");

//   // Load groundtruth for hybrid search.
//   vector<vector<labeltype>> hybrid_topks(nq);
//   load_hybrid_query_gt(c, l_bound, u_bound, k, hybrid_topks);
//   fmt::print("Finished loading groundtruth.\n");

//   // Compute selectivity.
//   int nsat;
//   stat_selectivity(attrs, l_bound, u_bound, nsat);

//   int m = 8;       // the number of dimensions per group
//   int nbits = 10;  // the number of bits to represent the sub-centroid
//   CompassIVF<float, float> comp(d, nb, nlist, nprobe, xb);

//   auto train_ivf_start = high_resolution_clock::now();
//   comp.TrainIvf(nb, xb);
//   auto train_ivf_stop = high_resolution_clock::now();
//   fmt::print(
//       "Finished training IVF, took {} microseconds.\n",
//       duration_cast<microseconds>(train_ivf_stop - train_ivf_start).count()
//   );

//   auto build_index_start = high_resolution_clock::now();
//   for (int i = 0; i < nb; i++) comp.Add(xb + i * d, i, attrs[i]);
//   auto build_index_stop = high_resolution_clock::now();
//   fmt::print(
//       "Finished building Compass, took {} microseconds.\n",
//       duration_cast<microseconds>(build_index_stop - build_index_start).count()
//   );

//   fmt::print("Finished adding points\n");

//   // statistics
//   vector<float> rec_at_ks(nq);
//   vector<float> pre_at_ks(nq);
//   vector<float> ivf_ppsl_qlty(nq, 0);
//   vector<float> ivf_ppsl_rate(nq, 0);
//   vector<float> perc_of_ivf_ppsl_in_tp(nq, 0);
//   vector<float> perc_of_ivf_ppsl_in_rz(nq, 0);
//   vector<float> linear_scan_rate(nq, 0);
//   vector<int> ivf_ppsl_nums(nq, 0);

//   for (int j = 0; j < nq; j++) {
//     fmt::print("Query: {:d},\n", j);
//     vector<bool> is_ivf_ppsl(nb, false);
//     auto rz = comp.SearchKnn(xq + j * d, k, l_bound, u_bound, is_ivf_ppsl);
//     std::set<labeltype> rz_indices, gt_indices, rz_gt_interse;

//     fmt::print("\tResult      : ");
//     int ivf_ppsl_in_rz = 0;
//     for (auto pair : rz) {
//       auto i = pair.second;
//       auto d = pair.first;
//       if (rz_indices.find(i) != rz_indices.end()) {
//         fmt::print("Found duplicate item {:d} in result indices.\n", i);
//       }
//       rz_indices.insert(i);
//       // fmt::print("\t\tLabel {:6d}, Dist {:9.2f}, Attr {:6.2f}\n", i, d, attrs[i]);
//       if (is_ivf_ppsl[i]) ivf_ppsl_in_rz++;
//     }
//     fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", rz.front().first, rz.back().first);

//     fmt::print("\tGround Truth: ");
//     int ivf_ppsl_in_tp = 0;
//     for (auto i : hybrid_topks[j]) {
//       // auto i = pair.second;
//       // auto d = pair.first;
//       gt_indices.insert(i);
//       // fmt::print("\t\tLabel {:6d}, Dist {:9.2f}, Attr {:6.2f}\n", i, d, attrs[i]);
//       if (is_ivf_ppsl[i]) ivf_ppsl_in_tp++;
//     }
//     auto gt_min = dist_func(xq + j * d, xb + hybrid_topks[j].front() * d, &d);
//     auto gt_max = dist_func(xq + j * d, xb + hybrid_topks[j].back() * d, &d);
//     fmt::print("Min: {:9.2f}, Max: {:9.2f}\n", gt_min, gt_max);

//     std::set_intersection(
//         gt_indices.begin(),
//         gt_indices.end(),
//         rz_indices.begin(),
//         rz_indices.end(),
//         std::inserter(rz_gt_interse, rz_gt_interse.begin())
//     );
//     rec_at_ks[j] = (float)rz_gt_interse.size() / gt_indices.size();
//     pre_at_ks[j] = (float)rz_gt_interse.size() / rz_indices.size();
//     fmt::print("\tRecall: {:5.2f}%, ", rec_at_ks[j] * 100);
//     fmt::print("Precision: {:5.2f}%, ", pre_at_ks[j] * 100);
//     fmt::print("{:3d}/{:3d}/{:3d}\n", rz_gt_interse.size(), rz_indices.size(), k);

//     auto num_ivf_ppsl = std::accumulate(is_ivf_ppsl.begin(), is_ivf_ppsl.end(), 0);
//     ivf_ppsl_qlty[j] = (float)ivf_ppsl_in_tp / num_ivf_ppsl;
//     ivf_ppsl_rate[j] = (float)ivf_ppsl_in_rz / num_ivf_ppsl;
//     perc_of_ivf_ppsl_in_tp[j] = (float)ivf_ppsl_in_tp / rz_gt_interse.size();
//     perc_of_ivf_ppsl_in_rz[j] = (float)ivf_ppsl_in_rz / rz.size();
//     linear_scan_rate[j] = (float)num_ivf_ppsl / nsat;
//     ivf_ppsl_nums[j] = num_ivf_ppsl;
//     fmt::print("\tNo. IVF Ppsl        : {:3d}\n", num_ivf_ppsl);
//     fmt::print("\tIVF Proposal Rate   : {:3d}/{:3d}\n", ivf_ppsl_in_rz, num_ivf_ppsl);
//     fmt::print("\tIVF Proposal Quality: {:3d}/{:3d}\n", ivf_ppsl_in_tp, num_ivf_ppsl);
//     fmt::print("\tIVF Proposal% in TP : {:3d}/{:3d}\n", ivf_ppsl_in_tp, rz_gt_interse.size());
//     fmt::print("\tIVF Proposal% in RZ : {:3d}/{:3d}\n", ivf_ppsl_in_rz, rz.size());
//     fmt::print("\tLinear Scan Rate    : {:3d}/{:3d}\n", num_ivf_ppsl, nsat);
//   }

//   fmt::print("Selectivity       : {}/{} = {:5.2f}%\n", nsat, nb, (double)nsat / nb * 100);
//   collate_compass_stats(
//       rec_at_ks,
//       pre_at_ks,
//       ivf_ppsl_qlty,
//       ivf_ppsl_rate,
//       perc_of_ivf_ppsl_in_tp,
//       perc_of_ivf_ppsl_in_rz,
//       linear_scan_rate,
//       ivf_ppsl_nums,
//       (log_dir / out_json).string()
//   );
// }