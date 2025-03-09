cmake_minimum_required(VERSION 3.12)



link_libraries(UTILS)

add_executable(binary_to_ascii binary_to_ascii.cpp)
add_executable(compute_groundtruth compute_groundtruth.cpp)
add_executable(compute_filter_groundtruth compute_filter_groundtruth.cpp)
add_executable(generate_attributes generate_attributes.cpp)