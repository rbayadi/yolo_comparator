#include "yolo_comparator_node.hpp"

YoloDetectorNode::YoloDetectorNode() : rclcpp::Node("yolo_detector")
{
    std::string model1_path = "model/yolov5s.onnx";
    std::string model2_path = "model/yolo26s.onnx";

    inference_ = std::make_unique<YoloInference>(model1_path, model2_path);
    RCLCPP_INFO(this->get_logger(), "Model1 loaded: %s", model1_path.c_str());
    RCLCPP_INFO(this->get_logger(), "Model2 loaded: %s", model2_path.c_str());

    auto qos = rclcpp::SensorDataQoS();
    subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        "/CAM_FRONT/image_rect_compressed", qos,
        std::bind(&YoloDetectorNode::image_callback, this, std::placeholders::_1)
    );
    annotations_publisher1_ = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>(
    "/yolov5/image_annotations", qos);

    elapsed_time_publisher1_ = this->create_publisher<std_msgs::msg::Float32>(
    "/yolov5_elapsed_time_milli", qos);

    annotations_publisher2_ = this->create_publisher<foxglove_msgs::msg::ImageAnnotations>(
    "/yolo26/image_annotations", qos);

    elapsed_time_publisher2_ = this->create_publisher<std_msgs::msg::Float32>(
    "/yolo26_elapsed_time_milli", qos);
}

void YoloDetectorNode::image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
    cv::Mat frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (frame.empty()) return;

    // Prepare variables and call ONNX sessions
    std::vector<float> yolov5_output;
    std::vector<float> yolo26_output;
    float yolov5_inference_time;
    float yolo26_inference_time;
    inference_->run(frame, 
        yolov5_output, 
        yolo26_output, 
        yolov5_inference_time, 
        yolo26_inference_time);

    // Next step is NMS + box scaling using the actual frame dimensions
    auto t0 = std::chrono::high_resolution_clock::now();
    auto detection_vector1 = postprocess_yolov5_output(yolov5_output, frame.cols, frame.rows);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto detection_vector2 = postprocess_yolo26_output(yolo26_output, frame.cols, frame.rows);
    auto t2 = std::chrono::high_resolution_clock::now();

    double postprocess_time1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double postprocess_time2 = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Publish bounding boxes
    foxglove_msgs::msg::Color model1_color;
    model1_color.r = 0.0;
    model1_color.g = 1.0;
    model1_color.b = 0.0;
    model1_color.a = 1.0;
    publish_annotations(annotations_publisher1_, detection_vector1, model1_color, msg->header);
    foxglove_msgs::msg::Color model2_color;
    model2_color.r = 1.0;
    model2_color.g = 0.0;
    model2_color.b = 0.0;
    model2_color.a = 1.0;
    publish_annotations(annotations_publisher2_, detection_vector2, model2_color, msg->header);

    // Publish elapsed time
    std_msgs::msg::Float32 elapsed_time_msg1;
    elapsed_time_msg1.data = yolov5_inference_time + postprocess_time1;
    elapsed_time_publisher1_->publish(elapsed_time_msg1);    

    std_msgs::msg::Float32 elapsed_time_msg2;
    elapsed_time_msg2.data = yolo26_inference_time + postprocess_time2;
    elapsed_time_publisher2_->publish(elapsed_time_msg2);    
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YoloDetectorNode>());
  rclcpp::shutdown();
  return 0;
}

