cmake_minimum_required(VERSION 3.14)
project(vo_frontend CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vo_core INTERFACE)
target_include_directories(vo_core INTERFACE ${CMAKE_SOURCE_DIR}/include)
