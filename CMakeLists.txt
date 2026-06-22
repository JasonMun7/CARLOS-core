cmake_minimum_required(VERSION 3.18)
project(carlos_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -march=native -Wall -Wextra)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# OpenMP on Linux; std::thread fallback on macOS to avoid libomp clash with PyTorch.
if(APPLE)
  message(STATUS "macOS: using std::thread parallelism (PyTorch libomp compatibility)")
else()
  find_package(OpenMP REQUIRED)
  add_compile_definitions(CARLOS_USE_OPENMP)
endif()

pybind11_add_module(_carlos_sim src/bindings.cpp)
target_include_directories(_carlos_sim PRIVATE src)

if(NOT APPLE)
  target_link_libraries(_carlos_sim PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _carlos_sim LIBRARY DESTINATION carlos)
