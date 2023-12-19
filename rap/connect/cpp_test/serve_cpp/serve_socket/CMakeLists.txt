cmake_minimum_required(VERSION 3.5)

project(serve_socket LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5Core)


add_executable(serve_socket
  main.cpp "include/timer.h"
  "include/platform.h" "src/platform.cpp"
)
target_link_libraries(serve_socket Qt5::Core)
target_link_libraries(serve_socket pthread)

