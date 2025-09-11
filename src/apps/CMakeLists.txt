cmake_minimum_required(VERSION 3.12)


link_libraries(UTILS)

add_executable(binary-to-ascii binary-to-ascii.cpp)
add_executable(compute-groundtruth compute-groundtruth.cpp)
add_executable(compute-filter-groundtruth compute-filter-groundtruth.cpp)
add_executable(generate-attributes generate-attributes.cpp)
add_executable(top-k-in-cluster-hist top-k-in-cluster-hist.cpp)
add_executable(top-k-in-cluster-hist-proclus top-k-in-cluster-hist-proclus.cpp)
add_executable(top-k-in-cluster-hist-pca top-k-in-cluster-hist-pca.cpp)
add_executable(cluster-imbalance-factor cluster-imbalance-factor.cpp)
add_executable(cluster-imbalance-factor-pca cluster-imbalance-factor-pca.cpp)
add_executable(prepare-for-navix prepare-for-navix.cpp)

target_link_libraries(top-k-in-cluster-hist Boost::filesystem)
target_link_libraries(top-k-in-cluster-hist-proclus Boost::filesystem)
target_link_libraries(top-k-in-cluster-hist-pca Boost::filesystem)
target_link_libraries(cluster-imbalance-factor Boost::filesystem)
target_link_libraries(cluster-imbalance-factor-pca Boost::filesystem)
target_link_libraries(prepare-for-navix Boost::filesystem)