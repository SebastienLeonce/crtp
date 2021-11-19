cmake_minimum_required(VERSION 3.8.0)
project(tp1 VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(crtp et.cpp)
add_compile_options(/std:c++20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
