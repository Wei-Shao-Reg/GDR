cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set (CMAKE_CXX_STANDARD 11)

project(GDR)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(GDR2D GDR2D.cxx)
add_executable(GDR3D GDR3D.cxx)

target_link_libraries(GDR2D ${ITK_LIBRARIES})
target_link_libraries(GDR3D ${ITK_LIBRARIES})

