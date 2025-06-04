#include "Compass1d.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class Compass1dX : public Compass1d<dist_t, attr_t> {
 protected:
  int dx_;

 public:
  Compass1dX(size_t n, size_t d, size_t dx, size_t M, size_t efc, size_t nlist)
      : Compass1d<dist_t, attr_t>(n, d, M, efc, nlist), dx_(dx) {}
};