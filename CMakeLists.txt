cmake_minimum_required(VERSION 3.9)

project( Naive_stereo_estimation )
project( DP_stereo_estimation )

###############
# packages #
###############

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

###############
# executables #
###############

add_executable( OpenCV_naive_stereo src/Naive.cpp )
target_link_libraries( OpenCV_naive_stereo ${OpenCV_LIBS} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenCV_naive_stereo OpenMP::OpenMP_CXX)
endif()

add_executable( OpenCV_DP_stereo src/Dynamic_programing.cpp )
target_link_libraries( OpenCV_DP_stereo ${OpenCV_LIBS} )
if(OpenMP_CXX_FOUND)
    target_link_libraries(OpenCV_DP_stereo OpenMP::OpenMP_CXX)
endif()



