file(GLOB_RECURSE native_srcs *.cc *.cpp)
add_library(UTILS STATIC ${native_srcs})