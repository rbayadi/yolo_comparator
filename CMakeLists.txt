cmake_minimum_required(VERSION 3.20)
project(yolo_comparator)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(foxglove_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
)

add_executable(yolo_comparator_node
include/detection_types.hpp
include/yolov5_post_processing.hpp
include/yolo26_post_processing.hpp
include/annotations_publishing.hpp
include/yolo_inference.hpp
include/yolo_comparator_node.hpp
src/yolov5_post_processing.cpp
src/yolo26_post_processing.cpp
src/annotations_publishing.cpp
src/yolo_inference.cpp
src/yolo_comparator_node.cpp)

target_include_directories(yolo_comparator_node PUBLIC
  include/
)

# ROS2 + OpenCV deps go through ament
ament_target_dependencies(yolo_comparator_node
  rclcpp
  std_msgs
  sensor_msgs
  foxglove_msgs
  vision_msgs
  cv_bridge
  OpenCV
)

# onnxruntime is an IMPORTED target — must use target_link_libraries directly
target_link_libraries(yolo_comparator_node
  onnxruntime
)

# also add your include/ folder for yolo_inference.hpp
target_include_directories(yolo_comparator_node PRIVATE
  include
)

install(TARGETS yolo_comparator_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
