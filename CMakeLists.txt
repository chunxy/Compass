cmake_minimum_required(VERSION 3.12)
project(Compass)
set(CMAKE_CXX_STANDARD 17)

message(STATUS ${CMAKE_CXX_FLAGS})
message(${CMAKE_SYSTEM})

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx" AVX_SUPPORTED)
check_cxx_compiler_flag("-mavx2" AVX2_SUPPORTED)

if(AVX2_SUPPORTED)
    message(STATUS "AVX2 supported")
endif()

if(AVX_SUPPORTED)
    message(STATUS "AVX supported")
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cmake_policy(SET CMP0167 NEW)
find_package(Boost REQUIRED COMPONENTS program_options context filesystem)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    set (LD_LIBRARY_PATH ${BOOST_LIBRARYDIR})
endif()
message(STATUS ${Boost_LIBRARIES})

find_package(fmt REQUIRED)

set(FAISS_ENABLE_GPU OFF)
set(FAISS_ENABLE_PYTHON OFF)
# add_compile_definitions(-DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF)

# include(FetchContent)
# FetchContent_Declare(
#   faiss
#   GIT_REPOSITORY https://github.com/facebookresearch/faiss
# )
# FetchContent_MakeAvailable(faiss)
add_subdirectory(thirdparty/faiss EXCLUDE_FROM_ALL)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/faiss)
link_libraries(faiss)

# add_subdirectory(thirdparty/SPTAG)
# include_directories(${PROJECT_SOURCE_DIR}/thirdparty/SPTAG)
# include_directories(${PROJECT_SOURCE_DIR}/thirdparty/SPTAG/AnnService)
# link_libraries(SPTAGLib)

add_subdirectory(thirdparty/btree EXCLUDE_FROM_ALL)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/btree)

# add_subdirectory(thirdparty/DiskANN)
# include_directories(${PROJECT_SOURCE_DIR}/thirdparty/DiskANN/include)
# link_libraries(diskann)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/hnswlib)
link_libraries(fmt::fmt-header-only Boost::program_options)

# option(NOSIMD "build hnsw on SIMD" ON)
# IF(NOSIMD)
#     add_definitions(-DNO_MANUAL_VECTORIZATION)
#     add_definitions(-DKGRAPH_NO_VECTORIZE)
# ENDIF(NOSIMD)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ftree-vectorize")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -DCOMPASS_DEBUG")

add_subdirectory(src)
