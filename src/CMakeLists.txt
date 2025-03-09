cmake_minimum_required(VERSION 3.12)



add_subdirectory(utils)
add_subdirectory(methods)
link_libraries(UTILS)
link_libraries(ACORN)
add_subdirectory(apps)
add_subdirectory(tests)
add_subdirectory(benchmarks)

