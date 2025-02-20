// #include <cstdint>
// #include "index.h"

// using namespace diskann;

// template <typename T, typename TagT = uint32_t, typename LabelT = uint32_t>
// class CompassNsg : public diskann::Index<T, TagT, LabelT> {
//  public:
//   std::pair<uint32_t, uint32_t> iterate_to_fixed_point(
//       InMemQueryScratch<T> *scratch,
//       const uint32_t Lindex,
//       const std::vector<uint32_t> &init_ids,
//       bool use_filter,
//       const std::vector<LabelT> &filters,
//       bool search_invocation
//   ) override {}

//   template <typename IdType>
//   std::pair<uint32_t, uint32_t>
//   search(const T *query, const size_t K, const uint32_t L, IdType *indices, float *distances) {
//     if (K > (uint64_t)L) {
//       throw ANNException("Set L to a value of at least K", -1, __FUNCSIG__, __FILE__, __LINE__);
//     }

//     ScratchStoreManager<InMemQueryScratch<T>> manager(_query_scratch);
//     auto scratch = manager.scratch_space();

//     if (L > scratch->get_L()) {
//       diskann::cout << "Attempting to expand query scratch_space. Was created "
//                     << "with Lsize: " << scratch->get_L() << " but search L is: " << L << std::endl;
//       scratch->resize_for_new_L(L);
//       diskann::cout << "Resize completed. New scratch->L is " << scratch->get_L() << std::endl;
//     }

//     const std::vector<LabelT> unused_filter_label;
//     const std::vector<uint32_t> init_ids = get_init_ids();

//     std::shared_lock<std::shared_timed_mutex> lock(_update_lock);

//     _data_store->preprocess_query(query, scratch);

//     auto retval = iterate_to_fixed_point(scratch, L, init_ids, false, unused_filter_label, true);

//     NeighborPriorityQueue &best_L_nodes = scratch->best_l_nodes();

//     size_t pos = 0;
//     for (size_t i = 0; i < best_L_nodes.size(); ++i) {
//       if (best_L_nodes[i].id < _max_points) {
//         // safe because Index uses uint32_t ids internally
//         // and IDType will be uint32_t or uint64_t
//         indices[pos] = (IdType)best_L_nodes[i].id;
//         if (distances != nullptr) {
// #ifdef EXEC_ENV_OLS
//           // DLVS expects negative distances
//           distances[pos] = best_L_nodes[i].distance;
// #else
//           distances[pos] = _dist_metric == diskann::Metric::INNER_PRODUCT ? -1 * best_L_nodes[i].distance
//                                                                           : best_L_nodes[i].distance;
// #endif
//         }
//         pos++;
//       }
//       if (pos == K) break;
//     }
//     if (pos < K) {
//       diskann::cerr << "Found pos: " << pos << "fewer than K elements " << K << " for query" << std::endl;
//     }

//     return retval;
//   }
// };

// int main(int argc, char **argv) { return 0; }