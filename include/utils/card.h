#pragma once

#include <cstdint>
#include <string>

struct DataCard {
  std::string name, base_path, query_path, groundtruth_path;
  uint32_t dim, n_base, n_queries, n_groundtruth;
  uint32_t attr_dim, attr_range;
  std::string attr_type;
};