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

# Silence '/*' within block comment warnings for GCC/Clang (incl. AppleClang)
# i.e. these
# DEBUG [ 66%] Building CXX object CMakeFiles/algotrading_v40_cpp.dir/src/cpp/feature_calculators/stochastic.cpp.o
# DEBUG /Users/chirayuagrawal/algotrading_v40/src/cpp/feature_calculators/stochastic.cpp:82:2: warning: '/*' within block comment [-Wcomment]
# DEBUG    82 | //*************************************************************
# DEBUG       |  ^
# DEBUG /Users/chirayuagrawal/algotrading_v40/src/cpp/feature_calculators/stochastic.cpp:84:2: warning: '/*' within block comment [-Wcomment]
# DEBUG    84 | //*************************************************************
# DEBUG       |  ^
# DEBUG /Users/chirayuagrawal/algotrading_v40/src/cpp/feature_calculators/stochastic.cpp:136:2: warning: '/*' within block comment [-Wcomment]
# DEBUG   136 | //*************************************************************
# DEBUG       |  ^
# DEBUG 3 warnings generated.

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(algotrading_v40_cpp PRIVATE -Wno-comment)
endif()

# Install the module
install(TARGETS algotrading_v40_cpp DESTINATION .)