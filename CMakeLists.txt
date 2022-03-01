cmake_minimum_required(VERSION 3.2)
project(nnpp)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug)
endif()

if(CMAKE_BUILD_TYPE EQUAL Release)
	add_definitions(-D_NDEBUG)
endif()

include_directories(source)
add_subdirectory(tests)
