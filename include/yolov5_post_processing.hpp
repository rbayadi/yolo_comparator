#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>
#include "detection_types.hpp"

std::vector<Detection> postprocess_yolov5_output(
    const std::vector<float>& yolo_output,
    int img_width,
    int img_height,
    float conf_threshold = 0.25f,
    float nms_threshold = 0.45f
);