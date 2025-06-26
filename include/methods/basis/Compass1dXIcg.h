#include <cstddef>
#include "Compass1dIcg.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass1dXIcg : public Compass1dIcg<dist_t, attr_t> {
 protected:
  int dx_;

  IterativeSearchState<dist_t> *Open(const dist_t *query, int idx, int nprobe) override {
    return this->isearch_->Open(query + idx * dx_, nprobe);
  }

 public:
  // This index only loads the ReentrantHnsw but does not build it.
  Compass1dXIcg(
      size_t n,
      size_t d,
      size_t dx,
      size_t M,
      size_t efc,
      size_t nlist,
      const string &path,
      size_t batch_k,
      size_t delta_efs
  )
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist), dx_(dx) {
    this->isearch_ = new IterativeSearch<dist_t>(n, dx, path, batch_k, delta_efs);
  }
};