cmake_minimum_required(VERSION 3.12)


file(GLOB srcs CONFIGURE_DEPENDS "*.cpp")
foreach(src ${srcs})
  get_filename_component(name ${src} NAME_WLE)
  # if (name STREQUAL "test_compass_1d")
  #   continue()
  # endif()
  add_executable("${name}" "${name}.cpp")
endforeach(src ${srcs})


target_link_libraries(test PRIVATE Boost::context)
target_link_libraries(test-coroutine PRIVATE Boost::context)

target_link_libraries(test-hnsw PRIVATE Boost::filesystem)
target_link_libraries(test-rng PRIVATE Boost::filesystem)
target_link_libraries(test-reentrant-hnsw PRIVATE Boost::filesystem)
target_link_libraries(test-iterative-hnsw PRIVATE Boost::filesystem)
target_link_libraries(test-iterative-hnsw-batch PRIVATE Boost::filesystem)
target_link_libraries(test-iterative-hnsw-neo PRIVATE Boost::filesystem)
target_link_libraries(test-iterative-hnsw-batch-neo PRIVATE Boost::filesystem)
target_link_libraries(test-iterative-hnsw-against-gt PRIVATE Boost::filesystem)

find_library(SIMDCompressionAndIntersection ${PROJECT_SOURCE_DIR}/thirdparty)
find_library(LIBBIT NAMES SIMDCompressionAndIntersection HINTS ${PROJECT_SOURCE_DIR}/thirdparty)
target_link_libraries(test-intersection PRIVATE ${LIBBIT})
target_link_libraries(test-bitset PRIVATE roaring)
