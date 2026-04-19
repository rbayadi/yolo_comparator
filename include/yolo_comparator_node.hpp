#include "annotations_publishing.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/points_annotation.hpp>

#include "yolo_inference.hpp"
#include "yolov5_post_processing.hpp"
#include "yolo26_post_processing.hpp"

#include <chrono>

class YoloDetectorNode : public rclcpp::Node
{
public:
  YoloDetectorNode();

private:
  void image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

  std::unique_ptr<YoloInference> inference_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
  rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_publisher1_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr elapsed_time_publisher1_;
  rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr annotations_publisher2_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr elapsed_time_publisher2_;
};
