#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>
#include <foxglove_msgs/msg/text_annotation.hpp>

#include "yolov5_post_processing.hpp"
#include "yolo26_post_processing.hpp"

void publish_annotations(
    const rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr publisher,
    const std::vector<Detection>& detections,
    const foxglove_msgs::msg::Color annot_color,
    const std_msgs::msg::Header& header);