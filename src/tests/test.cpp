
// #include <fmt/chrono.h>
// #include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <boost/coroutine2/all.hpp>
#include <boost/program_options.hpp>
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace po = boost::program_options;
using namespace std;

void foo(boost::coroutines2::coroutine<void>::push_type &sink) {
  fmt::print("start coroutine\n");
  sink();
  fmt::print("Point A\n");
  sink();
  fmt::print("finish coroutine\n");
}

template <typename T>
constexpr bool is_lvalue(T &&) {
  return std::is_lvalue_reference<T>{};
}

class Bar {
 public:
  int a;

 public:
  Bar(int a) : a(a) {}
  Bar(const Bar &other) = delete;
  Bar &operator=(const Bar &other) = delete;
};

Bar bar() { return Bar(10); }

int main(int argc, char **argv) {
  vector<float> l_bounds, u_bounds;
  uint32_t k;
  po::options_description configs;
  // configs.add_options()("l", po::value<decltype(l_bounds)>(&l_bounds)->required()->multitoken());
  // configs.add_options()("r", po::value<decltype(u_bounds)>(&u_bounds)->required()->multitoken());
  // configs.add_options()("k", po::value<decltype(k)>(&k)->required());
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, configs), vm);
  po::notify(vm);
  fmt::print("{}\n", l_bounds);
  fmt::print("{}\n", u_bounds);

  // set intersection
  set<int> set1 = {30, 20, 10, 40, 50, 60};
  set<int> set2 = {10, 30, 50, 40, 60, 80};
  set<int> result;
  set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(), inserter(result, result.begin()));

  // time_t ts = time(nullptr);
  // auto tm = localtime(&ts);
  // std::string log_file = fmt::format("{:%Y-%m-%d %H:%M:%S}", *tm);
  // fmt::print("{:%Y-%m-%d-%H-%M-%S}\n", *tm);
  // cout << asctime(tm) << '\n';
  // cout << log_file;

  string part = "/workspace/sift.ivecs";
  auto name = part.substr(0, part.find("."));
  name = name.substr(part.find_last_of("/") + 1, string::npos);

  using coroutine_t = boost::coroutines2::coroutine<void>;
  auto foo_lambda = [](coroutine_t::push_type &push) {
    fmt::print("Enter into the coroutine\n");
    push();
    fmt::print("Point A\n");
    push();
    fmt::print("End\n");

    vector<int> vec;
    for (int i = 0; i < 10000; i++) {
      vec.push_back(i);
    }
  };

  auto bar_lambda = [](coroutine_t::push_type &push) {
    fmt::print("Enter into the coroutine\n");
    push();

    // vector<int> vec;
    // for (int i = 0; i < 10000; i++) {
    //   vec.push_back(i);
    // }
  };

  coroutine_t::pull_type coroutine(foo_lambda);

  // map<int, coroutine_t::pull_type> map // constructs pair<int, coroutine_t::pull_type> with pair<int,
  // coroutine_t::pull_type&>
  map<int, coroutine_t::pull_type &> map;
  map.emplace(1, coroutine);

  for (int t = 0; t < 8; t++) {
    vector<coroutine_t::pull_type> functions;
    vector<int> numbers;
    for (int i = 0; i < 100; i++) {
      functions.emplace_back(std::move(foo_lambda));
      // numbers.push_back(i);
    }
  }
#pragma omp parallel for num_threads(4)
  for (int t = 0; t < 8; t++) {
    vector<coroutine_t::pull_type> functions;
    vector<int> numbers;
    for (int i = 0; i < 100; i++) {
      functions.emplace_back(std::move(foo_lambda));
      // numbers.push_back(i);
    }
  }

  fmt::print("1\n");
  coroutine();
  fmt::print("2\n");
  coroutine();
  fmt::print("3\n");

  fmt::print("{}\n", bar().a);
  fmt::print("is_lvalue: {}\n", is_lvalue(bar()));

  return 0;
}