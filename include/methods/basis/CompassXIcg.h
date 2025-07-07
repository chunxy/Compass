#include <cstddef>
#include "CompassIcg.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassXIcg : public CompassIcg<dist_t, attr_t> {
 protected:
  int dx_;

 public:
  // This index only loads the ReentrantHnsw but does not build it.
  CompassXIcg(
      size_t n,
      size_t d,
      size_t dx,
      SpaceInterface<dist_t> *s,
      size_t M,
      size_t efc,
      size_t nlist,
      size_t M_cg,
      size_t batch_k,
      size_t delta_efs
  )
      : CompassIcg<dist_t, attr_t>(n, d, s, M, efc, nlist, M_cg, batch_k, delta_efs), dx_(dx) {
    // NOTE: double allocation
    this->isearch_ = new IterativeSearch<dist_t>(n, dx, s, M_cg);
    this->isearch_->SetSearchParam(batch_k, delta_efs);
  }

  void LoadClusterGraph(fs::path path) override { this->isearch_->hnsw_->loadIndex(path.string(), new L2Space(dx_)); }
};