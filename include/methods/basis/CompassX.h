#include "Compass.h"

using hnswlib::labeltype;
using std::pair;
using std::priority_queue;
using std::vector;

template <typename dist_t, typename attr_t>
class CompassX : public Compass<dist_t, attr_t> {
 protected:
  int dx_;

 public:
  CompassX(size_t n, size_t d, size_t dx, size_t da, size_t M, size_t efc, size_t nlist)
      : Compass<dist_t, attr_t>(n, d, da, M, efc, nlist), dx_(dx) {}
};