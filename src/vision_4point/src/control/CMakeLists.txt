add_executable(control_node control_node.cc)

find_package(serial REQUIRED)
message("${serial_INCLUDE_DIRS}")
message("${serial_LIBRARIES}")

target_include_directories(control_node
PRIVATE
${serial_INCLUDE_DIRS}
   )

target_link_libraries(control_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${serial_LIBRARIES}
)



# target_include_directories(${PROJECT_NAME}
# PRIVATE
#     /usr/local/include/serial/serial.h
#     /usr/local/include/serial/v8stdint.h
#    )
# target_link_libraries(${PROJECT_NAME}
#     /usr/local/lib/libserial.so
#     )

