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
target_link_libraries(test_coroutine PRIVATE Boost::context)

target_link_libraries(test_hnsw PRIVATE Boost::filesystem)
target_link_libraries(test_rng PRIVATE Boost::filesystem)
target_link_libraries(test_reentrant_hnsw PRIVATE Boost::filesystem)
