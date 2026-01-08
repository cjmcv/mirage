cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
message(STATUS "CMake Version: ${CMAKE_VERSION}")

set(USE_CUDA ON)
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  project(MIRAGE LANGUAGES C CXX CUDA)
else()
  message(STATUS "USE_CUDA is OFF, will not use CUDA")
  project(MIRAGE LANGUAGES C CXX)
endif()

#include directories
include_directories(${CMAKE_INCLUDE_PATH})
include_directories(${CMAKE_BINARY_DIR}) # to include protobuf headeer files
include_directories("include")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

#add_subdirectory(deps/cutlass)

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")
#set(CUTLASS_NVCC_ARCHS "80;86")

file(GLOB_RECURSE MIRAGE_SRCS
  src/*.cc
)

file(GLOB_RECURSE MIRAGE_CUDA_SRCS
  src/*.cu
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

#set CUDA version
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(deps/cutlass/CUDA.cmake)
  string(REPLACE "." ";" CUDA_VERSION_PARTS ${CUDA_VERSION})
  list(GET CUDA_VERSION_PARTS 0 CUDA_VERSION_MAJOR)
  list(GET CUDA_VERSION_PARTS 1 CUDA_VERSION_MINOR)
  list(APPEND CMAKE_CUDA_FLAGS "-D__CUDACC_VER_MAJOR__=${CUDA_VERSION_MAJOR} -D__CUDACC_VER_MINOR__=${CUDA_VERSION_MINOR}")
endif()

# TODO: Currently disable CUTLASS_ARCH_MMA_SM80_ENABLED flag
# since we target Triton to perform codegen
#list(APPEND CMAKE_CUDA_FLAGS "-DCUTLASS_ARCH_MMA_SM80_ENABLED")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  message("Build in Debug mode")
  set(CMAKE_CUDA_FLAGS "-O0 -g -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fopenmp -O0 -g -Wall -fPIC ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CUDA_FLAGS "-O2 -Xcompiler=-fPIC ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_CXX_FLAGS "-fopenmp -O2 -Wall -fPIC ${CMAKE_CXX_FLAGS}")
endif()

#set CUDA
if (NOT "${USE_CUDA}" STREQUAL "OFF")
  include(cmake/cuda.cmake)
  find_cuda(${USE_CUDA})
  if (CUDA_FOUND)
    list(APPEND MIRAGE_SRCS ${MIRAGE_CUDA_SRCS})
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_INCLUDE_DIR=" ${CUDA_INCLUDE_DIRS})
    list(APPEND MIRAGE_LINK_LIBS ${CUDA_CUDART_LIBRARY})
    list(APPEND MIRAGE_LINK_LIBS ${CUDA_CUDA_LIBRARY})
  else()
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif(CUDA_FOUND)
endif()

#set CUTLASS
set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/deps/cutlass)
include_directories(deps/cutlass/include)
include_directories(deps/cutlass/tools/util/include)

add_subdirectory(deps/json)
list(APPEND MIRAGE_LINK_LIBS nlohmann_json::nlohmann_json)

link_directories(${CMAKE_SOURCE_DIR}/build/release)

add_library(mirage_runtime ${MIRAGE_SRCS})

list(APPEND MIRAGE_LINK_LIBS "-lstdc++fs")

set_target_properties(mirage_runtime PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")

target_compile_features(mirage_runtime PUBLIC cxx_std_17)

target_compile_definitions(mirage_runtime PUBLIC MIRAGE_BACKEND_USE_CUDA)

target_link_libraries(mirage_runtime ${MIRAGE_LINK_LIBS})

target_include_directories(mirage_runtime PUBLIC ${PROJECT_SOURCE_DIR}/include)

install(TARGETS mirage_runtime LIBRARY DESTINATION lib)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
    DESTINATION .)