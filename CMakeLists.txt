cmake_minimum_required(VERSION 3.18)
project(algotrading_v40_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# pull in everything under src/cpp/ (recursively)
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS
     ${PROJECT_SOURCE_DIR}/src/cpp/*.cpp)

# build ONE shared object called algotrading_v40_cpp$(ext)
pybind11_add_module(algotrading_v40_cpp ${CPP_SOURCES})

# Install the module
install(TARGETS algotrading_v40_cpp DESTINATION .)