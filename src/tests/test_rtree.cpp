// https://gist.github.com/warrenrentlytics/c9a1836a40d4fcbba28a7e29357dad7d

#include <fmt/core.h>
#include <fmt/format.h>
#include <boost/foreach.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/predicates.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

// namespace geo = boost::geometry;
namespace bgi = boost::geometry::index;
namespace geo = boost::geometry;
using point = geo::model::point<double, 2, geo::cs::cartesian>;
using box = geo::model::box<point>;
using value = std::pair<point, unsigned>;
using rtree = geo::index::rtree<value, geo::index::quadratic<16>>;

// typedef geo::model::point<double, 2, geo::cs::cartesian> point;
typedef std::pair<point, unsigned> value;

// template void point::set<0>(const double&);
// template void point::set<1>(const double&);

int main(int argc, char *argv[]) {
  // int n = std::atoi(argv[1]);
  int n = 10;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10.0, 10.0);
  // bgi::rtree<value, bgi::quadratic<16>> rtree;
  rtree rtree;

  // create some values
  for (int i = 0; i < n; ++i) {
    // insert new value
    double x = dis(gen);
    double y = dis(gen);
    rtree.insert(std::make_pair(point(x, y), i));
  }

  auto start = std::chrono::high_resolution_clock::now();
  std::vector<value> result_n;
  result_n.reserve(15);

  box b(point(-20, -20), point(20, 20));

  // boost::geometry::dispatch::covered_by<point, box>;
  // a.apply(point(0,0), b, a);
  auto rel_beg = rtree.qbegin(geo::index::covered_by(b));
  auto rel_end = rtree.qend();
  std::vector<value> vec(rel_beg, rel_end);
  fmt::print("{}\n", vec.size());
  rtree.query(bgi::covered_by(b), std::back_inserter(result_n));
  // rtree.query(bgi::nearest(point(0, 0), 15), std::back_inserter(result_n));
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::microseconds elapsed_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::chrono::duration<double> diff = end - start;
  std::cout << "knn query point:" << std::endl;
  std::cout << geo::wkt<point>(point(0, 0)) << std::endl;
  std::cout << elapsed_microseconds.count() << "us\n";
  std::cout << "knn query result:" << std::endl;
  BOOST_FOREACH (value const &v, result_n)
    std::cout << geo::wkt<point>(v.first) << " - " << v.second << std::endl;

  return 0;
}