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
  int k = 100;
  int dx = 64;
  int M = 32;
  int efc = 100;
  int nlist = 1000;  // the number of coarse clusters
  int nsub = 4;
  int nbits = 4;
  int M_cg = 4;
  int batch_k = 100;
  int delta_efs = 50;
  vector<int> nrel = {100};  // the number of candidates proposed by IVF per round
  vector<int> efs = {100};
  vector<int> nprobe = {100};
  int nthread = 1;
  int batchsz = 100;
  bool fast = true;

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
    optional_configs.add_options()("dx", po::value<decltype(dx)>(&dx));
    optional_configs.add_options()("M", po::value<decltype(M)>(&M));
    optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
    optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
    optional_configs.add_options()("nsub", po::value<decltype(nsub)>(&nsub));
    optional_configs.add_options()("nbits", po::value<decltype(nbits)>(&nbits));
    optional_configs.add_options()("M_cg", po::value<decltype(M_cg)>(&M_cg));
    optional_configs.add_options()("batch_k", po::value<decltype(batch_k)>(&batch_k));
    optional_configs.add_options()("delta_efs", po::value<decltype(delta_efs)>(&delta_efs));
    // index search parameters
    optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs)->multitoken());
    optional_configs.add_options()("nprobe", po::value<decltype(nprobe)>(&nprobe)->multitoken());
    optional_configs.add_options()("nrel", po::value<decltype(nrel)>(&nrel)->multitoken());
    // system parameters
    optional_configs.add_options()("nthread", po::value<decltype(nthread)>(&nthread));
    optional_configs.add_options()("batchsz", po::value<decltype(batchsz)>(&batchsz));
    optional_configs.add_options()("fast", po::value<decltype(fast)>(&fast));
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
  int k = 10;
  int dx = 64;
  int M = 32;
  int efc = 200;
  int nlist = 1000;  // the number of coarse clusters
  int M_cg = 4;
  vector<int> efs = {100};
  vector<int> nprobe = {10};
  vector<int> nrel = {200};  // the number of candidates proposed by IVF per round
  int nthread = 1;
  int batchsz = 100;
  bool fast = true;
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
    optional_configs.add_options()("dx", po::value<decltype(dx)>(&dx));
    optional_configs.add_options()("M", po::value<decltype(M)>(&M));
    optional_configs.add_options()("efc", po::value<decltype(efc)>(&efc));
    optional_configs.add_options()("nlist", po::value<decltype(nlist)>(&nlist));
    optional_configs.add_options()("M_cg", po::value<decltype(M_cg)>(&M_cg));
    // index search parameters
    optional_configs.add_options()("efs", po::value<decltype(efs)>(&efs)->multitoken());
    optional_configs.add_options()("nprobe", po::value<decltype(nprobe)>(&nprobe)->multitoken());
    optional_configs.add_options()("nrel", po::value<decltype(nrel)>(&nrel)->multitoken());
    // system parameters
    optional_configs.add_options()("nthread", po::value<decltype(nthread)>(&nthread));
    optional_configs.add_options()("batchsz", po::value<decltype(batchsz)>(&batchsz));
    optional_configs.add_options()("fast", po::value<decltype(fast)>(&fast));
    // Merge required and optional configs.
    configs.add(required_configs).add(optional_configs);
    // Parse arguments.
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, configs), vm);
    po::notify(vm);
  }
};

struct QueryMetric {
  std::vector<bool> is_ivf_ppsl;
  std::vector<bool> is_graph_ppsl;
  std::vector<float> cand_dist;
  int nround;
  int ncomp;
  int ncluster;
  int nrecycled;

  QueryMetric(int nb)
      : is_ivf_ppsl(nb, false), is_graph_ppsl(nb, false), nround(0), ncomp(0), ncluster(0), nrecycled(0) {}
};

struct BatchMetric {
  std::vector<QueryMetric> qmetrics;
  int latency;
  int cluster_search_time;
  int cluster_search_ncomp;

  BatchMetric(int nq, int nb)
      : qmetrics(nq, QueryMetric(nb)), latency(0), cluster_search_time(0), cluster_search_ncomp(0) {}
};

struct Stat {
  // per-query results
  vector<float> rec_at_ks;
  vector<float> pre_at_ks;
  vector<long> tp_s, rz_s;
  vector<float> gt_min_s, gt_max_s, rz_min_s, rz_max_s;
  vector<long> ivf_ppsl_in_rz_s, ivf_ppsl_in_tp_s;
  vector<long> graph_ppsl_in_rz_s, graph_ppsl_in_tp_s;
  // per-query intermediates
  vector<long> ivf_ppsl_nums;
  vector<float> ivf_ppsl_qlty;
  vector<float> ivf_ppsl_rate;
  vector<long> graph_ppsl_nums;
  vector<float> graph_ppsl_qlty;
  vector<float> graph_ppsl_rate;
  vector<vector<float>> cand_dist;
  vector<float> perc_of_ivf_ppsl_in_tp;
  vector<float> perc_of_ivf_ppsl_in_rz;
  vector<float> linear_scan_rate;
  vector<long> num_computations;
  vector<long> num_rounds;
  vector<long> num_clusters;
  vector<long> num_recycled;
  // per-batch stat
  vector<long> latencies;
  vector<long> cluster_search_time;
  vector<long> cluster_search_ncomp;

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
        cand_dist(nq),
        perc_of_ivf_ppsl_in_tp(nq, 0),
        perc_of_ivf_ppsl_in_rz(nq, 0),
        linear_scan_rate(nq, 0),
        num_computations(nq, 0),
        num_rounds(nq, 0),
        num_clusters(nq, 0),
        num_recycled(nq, 0) {}
};