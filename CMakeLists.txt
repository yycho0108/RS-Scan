cmake_minimum_required(VERSION 2.8.3)
project(realsense2_camera)
#add_compile_options(-std=c++11)
add_definitions(-DGOOGLE_CUDA=1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

option(BUILD_WITH_OPENMP "Use OpenMP" OFF)
option(SET_USER_BREAK_AT_STARTUP "Set user wait point in startup (for debug)" OFF)

if(BUILD_WITH_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "\n\n OpenMP is missing!\n\n")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if(SET_USER_BREAK_AT_STARTUP)
	message("GOT FLAG IN CmakeLists.txt")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBPDEBUG")
endif()

find_package(CUDA 10.0 REQUIRED)
IF (CUDA_FOUND)
    include_directories("${CUDA_INCLUDE_DIRS}")
	#include_directories("/usr/local/cuda-10.0/targets/x86_64-linux/include/crt")
    # Optional: Specify the arch of your CUDA hardware here
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11 -gencode arch=compute_61,code=sm_61)
ENDIF ()

## Eigen3
find_package(Eigen3 REQUIRED)
if (EIGEN3_INCLUDE_DIR)
    include_directories("${EIGEN3_INCLUDE_DIR}")
endif (EIGEN3_INCLUDE_DIR)


find_package(realsense2 2.17.1 REQUIRED)
if(NOT realsense2_FOUND)
    message(FATAL_ERROR "\n\n Intel RealSense SDK 2.0 is missing, please install it from https://github.com/IntelRealSense/librealsense/releases\n\n")
endif()

find_package(OpenCV 4.0.1 REQUIRED)

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

SET( CMAKE_BUILD_TYPE Release FORCE )

#if (CMAKE_BUILD_TYPE EQUAL "RELEASE")
    message(STATUS "Create Release Build.")
    set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
    #else()
    #    message(STATUS "Create Debug Build.")
    #endif()

set(CMAKE_CXX_FLAGS "-fPIE -fPIC -std=c++11 -D_FORTIFY_SOURCE=2 -fstack-protector -Wformat -Wformat-security -Wall ${CMAKE_CXX_FLAGS}")

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Weffc++")

include_directories(
    include
    ${realsense_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
	"/home/jamie/libs/KinectFusionLib/include"
    )

link_directories("/home/jamie/libs/KinectFusionLib/build")


#add_executable(main main.cpp)
cuda_add_executable(main main.cpp src/tracker.cpp)
target_link_libraries(main
    ${realsense2_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS}
	KinectFusion
    ${PCL_LIBRARIES}
    )

add_executable(test_ir test_ir.cpp)
target_link_libraries(test_ir
    ${realsense2_LIBRARY}
    ${CMAKE_THREAD_LIBS_INIT}
	${OpenCV_LIBS}
    )
