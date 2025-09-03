  cmake_minimum_required(VERSION 3.14)
  project(vo_frontend CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  option(USE_PANGOLIN "Enable Pangolin visualizer" OFF)

  # Core include interface
  add_library(vo_core INTERFACE)
  target_include_directories(vo_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

  # Dependencies
  find_package(OpenCV REQUIRED)
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
  find_package(OpenGL REQUIRED)
  find_package(Python3 COMPONENTS Development NumPy REQUIRED)
  if(USE_PANGOLIN)
  find_package(Pangolin REQUIRED PATHS /usr/local/lib/cmake/Pangolin)
  endif()



  # -------------------
  # Test: detect + match
  # -------------------
  add_executable(test_detect_match
    src/apps/test_detect_match.cpp
    src/frontend/feature_detector.cpp
    src/frontend/feature_matcher.cpp
  )
  target_include_directories(test_detect_match PRIVATE include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_detect_match PRIVATE ${OpenCV_LIBS})

  # -------------------
  # Test: init + triangulate
  # -------------------
  add_executable(test_init_triangulate
    src/apps/test_init_triangulate.cpp
    src/frontend/feature_detector.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/initializer.cpp
    src/frontend/triangulator.cpp
  )
  target_include_directories(test_init_triangulate PRIVATE include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_init_triangulate PRIVATE ${OpenCV_LIBS})

  # -------------------
  # Test: PnP + track
  # -------------------
  add_executable(test_pnp_track
    src/apps/test_pnp_track.cpp
    src/frontend/feature_detector.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/initializer.cpp
    src/frontend/triangulator.cpp
    src/frontend/pose_estimator.cpp
  )
  target_include_directories(test_pnp_track PRIVATE include ${EIGEN3_INCLUDE_DIRS})
  target_link_libraries(test_pnp_track PRIVATE ${OpenCV_LIBS})

  # -------------------
  # Main frontend app
  # -------------------
  add_executable(run_frontend
    src/apps/run_frontend.cpp
    src/frontend/frontend.cpp
    src/frontend/feature_detector.cpp
    src/frontend/feature_matcher.cpp
    src/frontend/initializer.cpp
    src/frontend/triangulator.cpp
    src/frontend/pose_estimator.cpp
  )

  target_include_directories(run_frontend PRIVATE
      include
      ${EIGEN3_INCLUDE_DIRS}
      ${Python3_INCLUDE_DIRS}
      ${Python3_NumPy_INCLUDE_DIRS}
      ${CMAKE_SOURCE_DIR}/include/thirdparty
  )

  target_link_libraries(run_frontend PRIVATE
    ${OpenCV_LIBS}
    ${Python3_LIBRARIES}
    ${OPENGL_LIBRARIES}
)
if(USE_PANGOLIN)
  target_sources(run_frontend PRIVATE src/visualization/pangolin_visualizer.cpp)
  target_link_libraries(run_frontend PRIVATE
    pango_core
    pango_display
    pango_geometry
    pango_glgeometry
    pango_image
    pango_opengl
    pango_plot
    pango_scene
    pango_tools
    pango_vars
    pango_video
    pango_windowing
  )
endif()


