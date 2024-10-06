# 1.添加可执行文件
file( GLOB_RECURSE source_detector 
  ${CMAKE_SOURCE_DIR}/vision_4point/src/detector/*.cc
	${CMAKE_SOURCE_DIR}/vision_4point/src/detector/infer/*.cc
	${CMAKE_SOURCE_DIR}/vision_4point/src/detector/infer/*.cu
) 
add_executable(detector_node ${source_detector})


# 3.链接
target_link_libraries(detector_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  nvinfer nvinfer_plugin nvonnxparser nvparsers cudart cublas
)

# 4. 添加依赖项
add_dependencies(detector_node ${PROJECT_NAME}_generate_messages_cpp)

# 输出环境变量信息
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
