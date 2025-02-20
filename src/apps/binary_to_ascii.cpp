#include <fmt/core.h>
#include <fmt/format.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ios>
#include <ostream>
#include <string>
#include "config.h"
#include "utils/card.h"

namespace po = boost::program_options;

int main() {
  std::string dataname = "siftsmall_1_1000";
  po::options_description configs;
  configs.add_options()("datacard", po::value<decltype(dataname)>(&dataname));

  extern std::map<std::string, DataCard> name_to_card;
  DataCard c = name_to_card[dataname];
  for (auto part : {c.base_path, c.query_path, c.groundtruth_path}) {
    std::ifstream ifs(part, std::ios::binary | std::ios::in);
    std::ofstream ofs(
        DATA + "/" +
        part.substr(
            part.find_last_of("/") + 1, part.find(".") - part.find_last_of("/") - 1) +
        ".txt");
    uint32_t d;
    float *nums = new float[128];
    while (ifs.read((char *)&d, sizeof(d))) {
      assert(d == 128 || d == 100);
      ifs.read((char *)nums, 4 * d);
      for (size_t i = 0; i < d; i++) {
        if (boost::algorithm::ends_with(part, "groundtruth.ivecs"))
          ofs << fmt::format("{:6d}", *reinterpret_cast<uint32_t *>(nums + i));
        else
          ofs << fmt::format("{:6g}", nums[i]);
      }
      ofs << std::endl;
    }
  }

  return 0;
}