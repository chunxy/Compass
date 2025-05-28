cmake_minimum_required(VERSION 3.12)



add_subdirectory(utils)
link_libraries(UTILS)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(benchmarks)

