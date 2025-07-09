#include <cstdlib>
#include "hnswlib/hnswlib.h"

int main() {
  srand(123456);

  int dim_s[] = {128, 200, 209, 300};
  for (auto dim : dim_s) {
    hnswlib::L2SpaceB space(dim);
    uint8_t *x1 = new uint8_t[dim];
    uint8_t *x2 = new uint8_t[dim];

    for (int i = 0; i < dim; i++) {
      x1[i] = rand() % 256;
      x2[i] = rand() % 256;
    }

    auto dist = space.get_dist_func()(x1, x2, space.get_dist_func_param());

    int res = 0;
    for (int i = 0; i < dim; i++) {
      res += (int)(x1[i] - x2[i]) * (int)(x1[i] - x2[i]);
    }
    std::cout << "Distance: " << res << " as to " << dist << std::endl;
  }
}