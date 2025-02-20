#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;
namespace po = boost::program_options;

struct IvfGraph1dArgs {
  std::string datacard;
  float l_bound, u_bound;
  int k;
  int M;
  int efc;
  int nlist;  // the number of coarse clusters
  int nrel;   // the number of candidates proposed by IVF per round
  int efs;
  int min_comp = 1000;
  int nprobe;
  int nthread = 1;
  int batchsz = 100;

  IvfGraph1dArgs(int argc, char **argv) {
    po::options_description configs;
    po::options_description required_configs("Required"), optional_configs("Optional");
    // dataset parameter
    required_configs.add_options()("datacard", po::value<decltype(datacard)>(&datacard)->required());
    // search parameters
    required_configs.add_options()("l", po::value<decltype(l_bound)>(&l_bound)->required());
    required_configs.add_options()("r", po::value<decltype(u_bound)>(&u_bound)->required());
    required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
    // index constrcution parameters
    optional_configs.add_options()("M", po::value<decltype(M)>(&M));
    optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
    optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
    // index search parameters
    optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs));
    optional_configs.add_options()("nprobe", po::value<decltype(nprobe)>(&nprobe));
    optional_configs.add_options()("nrel", po::value<decltype(nrel)>(&nrel));
    optional_configs.add_options()("mincomp", po::value<decltype(min_comp)>(&min_comp));
    // system parameters
    optional_configs.add_options()("nthread", po::value<decltype(nthread)>(&nthread));
    optional_configs.add_options()("batchsz", po::value<decltype(batchsz)>(&batchsz));

    // Merge required and optional configs.
    configs.add(required_configs).add(optional_configs);
    // Parse arguments.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, configs), vm);
    po::notify(vm);
  }
};

struct IvfGraph2dArgs {
  std::string datacard;
  vector<float> l_bounds, u_bounds;
  int k;
  int M;
  int efc;
  int nlist;  // the number of coarse clusters
  int efs;
  int nprobe;
  int nrel;   // the number of candidates proposed by IVF per round
  int min_comp = 1000;
  int nthread = 1;
  int batchsz = 100;

  IvfGraph2dArgs(int argc, char **argv) {
    po::options_description configs;
    po::options_description required_configs("Required"), optional_configs("Optional");
    // dataset parameter
    required_configs.add_options()("datacard", po::value<decltype(datacard)>(&datacard)->required());
    // search parameters
    required_configs.add_options()("l", po::value<decltype(l_bounds)>(&l_bounds)->required()->multitoken());
    required_configs.add_options()("r", po::value<decltype(u_bounds)>(&u_bounds)->required()->multitoken());
    required_configs.add_options()("k", po::value<decltype(k)>(&k)->required());
    // index constrcution parameters
    optional_configs.add_options()("M", po::value<decltype(M)>(&M));
    optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
    optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
    // index search parameters
    optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs));
    optional_configs.add_options()("nprobe", po::value<decltype(nprobe)>(&nprobe));
    optional_configs.add_options()("nrel", po::value<decltype(nrel)>(&nrel));
    optional_configs.add_options()("mincomp", po::value<decltype(min_comp)>(&min_comp));
    // system parameters
    optional_configs.add_options()("nthread", po::value<decltype(nthread)>(&nthread));
    optional_configs.add_options()("batchsz", po::value<decltype(batchsz)>(&batchsz));

    // Merge required and optional configs.
    configs.add(required_configs).add(optional_configs);
    // Parse arguments.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, configs), vm);
    po::notify(vm);
  }
};

struct Metric {
  std::vector<bool> is_ivf_ppsl;
  std::vector<bool> is_graph_ppsl;
  long tcontext;
  int nround;
  int ncomp;

  Metric(int nb) : is_ivf_ppsl(nb, false), is_graph_ppsl(nb, false), nround(0), tcontext(0), ncomp(0) {}
};

struct Stat {
  // result
  vector<float> rec_at_ks;
  vector<float> pre_at_ks;
  vector<int> tp_s, rz_s;
  vector<float> gt_min_s, gt_max_s, rz_min_s, rz_max_s;
  vector<int> ivf_ppsl_in_rz_s, ivf_ppsl_in_tp_s;
  vector<int> graph_ppsl_in_rz_s, graph_ppsl_in_tp_s;
  // intermediate
  vector<int> ivf_ppsl_nums;
  vector<float> ivf_ppsl_qlty;
  vector<float> ivf_ppsl_rate;
  vector<int> graph_ppsl_nums;
  vector<float> graph_ppsl_qlty;
  vector<float> graph_ppsl_rate;
  vector<float> perc_of_ivf_ppsl_in_tp;
  vector<float> perc_of_ivf_ppsl_in_rz;
  vector<float> linear_scan_rate;
  vector<int> num_computations;
  vector<int> num_rounds;
  // system
  vector<long> latencies;
  vector<long> ctx_switch_time;

  Stat(int nq)
      : rec_at_ks(nq, 0),
        pre_at_ks(nq, 0),
        tp_s(nq, 0),
        rz_s(nq, 0),
        gt_min_s(nq, 0),
        gt_max_s(nq, 0),
        rz_min_s(nq, 0),
        rz_max_s(nq, 0),
        ivf_ppsl_in_rz_s(nq, 0),
        ivf_ppsl_in_tp_s(nq, 0),
        graph_ppsl_in_rz_s(nq, 0),
        graph_ppsl_in_tp_s(nq, 0),
        ivf_ppsl_nums(nq, 0),
        ivf_ppsl_qlty(nq, 0),
        ivf_ppsl_rate(nq, 0),
        graph_ppsl_nums(nq, 0),
        graph_ppsl_qlty(nq, 0),
        graph_ppsl_rate(nq, 0),
        perc_of_ivf_ppsl_in_tp(nq, 0),
        perc_of_ivf_ppsl_in_rz(nq, 0),
        linear_scan_rate(nq, 0),
        num_computations(nq, 0),
        num_rounds(nq, 0),
        latencies(nq, 0),
        ctx_switch_time(nq, 0) {}
};