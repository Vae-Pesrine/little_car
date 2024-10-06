add_executable(get_image_node get_image_node.cc)


target_link_libraries(get_image_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

