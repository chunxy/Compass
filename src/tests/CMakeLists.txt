cmake_minimum_required(VERSION 3.12)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADDITIONAL_CFLAGS}")

add_subdirectory(acorn)
# add_executable(test_compass_1d test_compass_1d.cpp)
# add_executable(test_ivfpq test_ivfpq.cpp)
# add_executable(test_rtree test_rtree.cpp)
# add_executable(test test.cpp)
# add_executable(test_coroutine test_coroutine.cpp)
# add_executable(test_coroutine_base test_coroutine_base.cpp)
# add_executable(test_omp test_omp.cpp)
file(GLOB srcs CONFIGURE_DEPENDS "*.cpp")
foreach(src ${srcs})
  get_filename_component(name ${src} NAME_WLE)
  if (name STREQUAL "test_compass_1d")
    continue()
  endif()
  add_executable("${name}" "${name}.cpp")
endforeach(src ${srcs})


# target_link_libraries(test_compass_1d PRIVATE Boost::context)
target_link_libraries(test PRIVATE Boost::context)
target_link_libraries(test_coroutine PRIVATE Boost::context)
target_link_libraries(test_coroutine_base PRIVATE Boost::context)
