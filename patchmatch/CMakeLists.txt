cmake_minimum_required (VERSION 2.8)
project (PATCHMATCH)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=gnu++11")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(.)

add_executable(
        patchmatch
        pm.h
        pm.cpp
        main.cpp
)
target_link_libraries(patchmatch
        ${OpenCV_LIBS}
        )
target_link_libraries(patchmatch  OpenMP::OpenMP_CXX)

