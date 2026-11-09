cmake_minimum_required(VERSION 3.18)
project(nash_mcts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(nash_mcts bindings.cpp)
target_compile_options(nash_mcts PRIVATE -O3 -Wall -Wextra)
# Put the extension next to this file so `cpp/mcts` is the import location.
set_target_properties(nash_mcts PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
