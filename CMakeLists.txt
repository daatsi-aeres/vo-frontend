cmake_minimum_required(VERSION 3.14)
project(vo_frontend CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vo_core INTERFACE)
target_include_directories(vo_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(test_detect_match
  src/apps/test_detect_match.cpp
  src/frontend/feature_detector.cpp
  src/frontend/feature_matcher.cpp)
target_include_directories(test_detect_match PRIVATE include ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(test_detect_match PRIVATE ${OpenCV_LIBS})

add_executable(test_init_triangulate
  src/apps/test_init_triangulate.cpp
  src/frontend/feature_detector.cpp
  src/frontend/feature_matcher.cpp
  src/frontend/initializer.cpp
  src/frontend/triangulator.cpp)

target_include_directories(test_init_triangulate PRIVATE include)
target_link_libraries(test_init_triangulate PRIVATE ${OpenCV_LIBS})
# if needed:
target_include_directories(test_init_triangulate PRIVATE ${EIGEN3_INCLUDE_DIRS})




