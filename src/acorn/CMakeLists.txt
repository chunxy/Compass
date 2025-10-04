file(GLOB_RECURSE native_srcs *.cc *.cpp)
add_library(ACORN STATIC ${native_srcs})

target_compile_definitions(ACORN PRIVATE FINTEGER=int)
