
// #include <fmt/chrono.h>
// #include <fmt/core.h>
// #include <fmt/format.h>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <boost/coroutine2/all.hpp>
#include <ctime>

using namespace std;

void foo(boost::coroutines2::coroutine<void>::push_type &sink) {
  fmt::print("start coroutine\n");
  sink();
  fmt::print("Point A\n");
  sink();
  fmt::print("finish coroutine\n");
}

int main(int argc, char **argv) {
  using coroutine_t = boost::coroutines2::coroutine<void>;
  auto foo_lambda = [](coroutine_t::push_type &push) {
    fmt::print("Enter into the coroutine\n");
    push();
    fmt::print("Point A\n");
    push();
    fmt::print("End\n");
  };

  coroutine_t::pull_type coroutine(foo_lambda);

  fmt::print("1\n");
  coroutine();
  fmt::print("2\n");
  coroutine();
  fmt::print("3\n");

  return 0;
}