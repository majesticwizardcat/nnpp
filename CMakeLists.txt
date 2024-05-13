cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(nnpp)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

if (${CMAKE_BUILD_TYPE} MATCHES "Release")
	message(STATUS "Buidling Release")
	add_compile_definitions("NDEBUG")
	add_compile_options(-O3)
else()
	message(STATUS "Buidling Debug")
endif()

include_directories(standard-sauce)
include_directories(pcg-cpp/include)
include_directories(source)
add_subdirectory(tests)

