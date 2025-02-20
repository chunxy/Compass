cmake_minimum_required(VERSION 3.12)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CFLAGS}")

link_libraries(UTILS)

add_executable(binary_to_ascii binary_to_ascii.cpp)
add_executable(diskann_compute_groundtruth compute_groundtruth.cpp)
add_executable(compute_filter_groundtruth compute_filter_groundtruth.cpp)
add_executable(diskann_generate_attributes generate_attributes.cpp)