#include <fmt/core.h>
#include <fmt/format.h>
#include <boost/program_options.hpp>
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include "config.h"
#include "utils/card.h"
#include "utils/reader.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
  std::string dataname;

  po::options_description configs;
  configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  std::string name = c.name;
  uint32_t n = c.n_base, attr_d = c.attr_dim, attr_range = c.attr_range;
  std::string type = c.attr_type;

  if (type == "float32") {
    std::string path = fmt::format(VALUE_PATH_TMPL, name, attr_d, attr_range);
    fmt::print("Saving to {}", path);
    BinaryAttrReader<float>::GenerateRandomAttrs(path, n, attr_d, attr_range);
    BinaryAttrReader<float> value_reader(path);
    auto values = value_reader.GetAttrs();
    for (auto value : values) {
      for (auto num : value) {
        fmt::print("{:5.2f} ", num);
      }
      fmt::print("\n");
    }
  } else if (type == "int32") {
    std::string blabel_path = fmt::format(BLABEL_PATH_TMPL, name, attr_range);
    fmt::print("Saving to {}", blabel_path);
    BinaryAttrReader<int32_t>::GenerateRandomAttrs(blabel_path, n, attr_d, attr_range);
    BinaryAttrReader<int32_t> blabel_reader(blabel_path);
    auto blabels = blabel_reader.GetAttrs();
    for (auto label : blabels) {
      for (auto num : label) {
        fmt::print("{:5d} ", num);
      }
      fmt::print("\n");
    }

    std::string qlabel_path = fmt::format(QLABEL_PATH_TMPL, name, attr_range);
    fmt::print("Saving to {}", qlabel_path);
    BinaryAttrReader<int32_t>::GenerateRandomAttrs(qlabel_path, n, attr_d, attr_range);
    BinaryAttrReader<int32_t> qlabel_reader(blabel_path);
    auto qlabels = qlabel_reader.GetAttrs();
    for (auto label : qlabels) {
      for (auto num : label) {
        fmt::print("{:5d} ", num);
      }
      fmt::print("\n");
    }
  }

  return 0;
}