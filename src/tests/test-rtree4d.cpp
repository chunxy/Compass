#include <iostream>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

int main() {
  // 1. Define a 4-dimensional point type
  // This uses double for coordinates, has 4 dimensions, and uses a Cartesian coordinate system.
  typedef bg::model::point<double, 4, bg::cs::cartesian> point4d;

  // 2. Declare the R-tree using the 4D point type
  // We are storing point4d objects and using a standard quadratic splitting algorithm.
  bgi::rtree<point4d, bgi::quadratic<16>> rtree;

  // 3. Insert some 4D data
  point4d p1(1.0, 2.0);
  p1.set<2>(3.0);
  p1.set<3>(4.0);
  point4d p2(5.0, 6.0);
  p2.set<2>(7.0);
  p2.set<3>(8.0);
  point4d p3(2.0, 4.0);
  p3.set<2>(6.0);
  p3.set<3>(8.0);
  rtree.insert(p1);
  rtree.insert(p2);
  rtree.insert(p3);


  // 4. Define a 4D query box
  // Find points where: 0<=attr1<=3, 1<=attr2<=5, 2<=attr3<=7, 3<=attr4<=9
  point4d lower_bound(0.0, 1.0);
  lower_bound.set<2>(2.0);
  lower_bound.set<3>(3.0);
  point4d upper_bound(3.0, 5.0);
  upper_bound.set<2>(6.5);
  upper_bound.set<3>(9.0);
  bg::model::box<point4d> query_box(lower_bound, upper_bound);

  // 5. Perform the query
  std::vector<point4d> result;
  rtree.query(bgi::within(query_box), std::back_inserter(result));

  // Print results
  std::cout << "Found " << result.size() << " points:" << std::endl;
  for (const auto &p : result) {
    std::cout << bg::dsv(p) << std::endl;
  }

  return 0;
}