cmake_minimum_required(VERSION 3.0.2)
project(zbar_test)
find_package(OpenCV 4.5.4 REQUIRED)
add_executable(zbar_test test.cc)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(zbar_test zbar ${OpenCV_LIBS})
