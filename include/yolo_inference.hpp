#pragma once

#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>

class YoloInference
{
public:
  YoloInference(const std::string & model1_path, const std::string & model2_path);

  // Returns raw output tensor as flat float vector
  void run(const cv::Mat & bgr_frame, 
    std::vector<float>& output1,
    std::vector<float>& output2,
    float& inference_time1, 
    float& inference_time2);

private:
  cv::Mat preprocess(const cv::Mat & bgr_frame);

  Ort::Env env_;
  Ort::SessionOptions session_options_;  // default: 1 thread, CPU EP
  Ort::Session session1_;
  Ort::Session session2_;
  Ort::MemoryInfo memory_info_;

  std::string input_name_;
  std::string output1_name_;
  std::string output2_name_;
  std::vector<int64_t> input_dims_;
};
