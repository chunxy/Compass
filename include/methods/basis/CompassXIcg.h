#include <cstddef>
#include <type_traits>
#include "CompassIcg.h"

using hnswlib::L2Space;
using hnswlib::L2SpaceB;
using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t, typename cg_dist_t = float>
class CompassXIcg : public CompassIcg<dist_t, attr_t, cg_dist_t> {
 protected:
  int dx_;

 public:
  // This index only loads the ReentrantHnsw but does not build it.
  CompassXIcg(
      size_t n,
      size_t d,
      size_t dx,
      SpaceInterface<cg_dist_t> *s,
      size_t da,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t delta_efs
  )
      : CompassIcg<dist_t, attr_t, cg_dist_t>(n, d, s, da, M, efc, nlist, M_cg, batch_k, delta_efs), dx_(dx) {
    if (this->isearch_) delete this->isearch_;
    this->isearch_ = new IterativeSearch<cg_dist_t>(n, dx, s, M_cg);
    this->isearch_->SetSearchParam(batch_k, delta_efs);
  }

  void LoadClusterGraph(fs::path path) override {
    using SpaceType = typename std::conditional<std::is_same<cg_dist_t, int>::value, L2SpaceB, L2Space>::type;
    this->isearch_->hnsw_->loadIndex(path.string(), new SpaceType(dx_));
  }
};