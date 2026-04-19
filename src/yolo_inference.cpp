#include "yolo_inference.hpp"

// This static function is used in the inference class constructor to define
// the session options. The values defined here play a key role in reducing the
// inference elapsed time!
static Ort::SessionOptions CreateSessionOptions() {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(6);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return opts;
}

// This static function is used to construct the memory_info_ member variable
// which will be allocated once and will be kept on using subsequently. This 
// also helps in reducing the inference elapsed time.
static Ort::MemoryInfo CreateMemoryInfo() {
    return Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

YoloInference::YoloInference(const std::string & model1_path, const std::string & model2_path)
: env_(ORT_LOGGING_LEVEL_WARNING, "yolo_comparator"),
  session_options_(CreateSessionOptions()),             
  session1_(env_, model1_path.c_str(), session_options_),
  session2_(env_, model2_path.c_str(), session_options_),
  memory_info_(CreateMemoryInfo())
{
  // Cache input/output names — ORT allocates these, we hold the strings
  Ort::AllocatorWithDefaultOptions allocator;

  // Input
  auto input_name = session1_.GetInputNameAllocated(0, allocator);
  input_name_ = std::string(input_name.get());

  // Output for model 1
  auto output1_name = session1_.GetOutputNameAllocated(0, allocator);
  output1_name_ = std::string(output1_name.get());

  // Output for model 2
  auto output2_name = session2_.GetOutputNameAllocated(0, allocator);
  output2_name_ = std::string(output1_name.get());

  // Verify expected input shape: [1, 3, 640, 640]
  auto input_shape_info = session1_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
  input_dims_ = input_shape_info.GetShape();  // {1, 3, 640, 640}
}

void YoloInference::run(const cv::Mat & bgr_frame, 
    std::vector<float>& output1,
    std::vector<float>& output2,
    float& inference_time1, 
    float& inference_time2)
{
  cv::Mat input_blob = preprocess(bgr_frame);

  // Build input tensor — data lives in input_blob, OrtMemoryInfo is CPU

  std::vector<int64_t> input_shape = {1, 3, 640, 640};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info_,
    reinterpret_cast<float *>(input_blob.data),
    input_blob.total() * input_blob.elemSize() / sizeof(float),
    input_shape.data(),
    input_shape.size()
  );

  const char * const input_names[]  = {input_name_.c_str()};
  const char * const output1_names[] = {output1_name_.c_str()};
  const char * const output2_names[] = {output2_name_.c_str()};

  auto t0 = std::chrono::high_resolution_clock::now();

  auto output1_tensors = session1_.Run(
    Ort::RunOptions{nullptr},
    input_names,  &input_tensor, 1,
    output1_names, 1
  );

  // YOLOv5 output shape: [1, 25200, 85]
  // 25200 anchor predictions, each = [x, y, w, h, obj_conf, 80 class scores]
  auto & out1 = output1_tensors[0];
  auto shape1 = out1.GetTensorTypeAndShapeInfo().GetShape();
  const float * data1 = out1.GetTensorData<float>();
  size_t count1 = shape1[1] * shape1[2];  // 300 * 6

  output1 = std::vector<float>(data1, data1 + count1);

  auto t1 = std::chrono::high_resolution_clock::now();

  auto output2_tensors = session2_.Run(
    Ort::RunOptions{nullptr},
    input_names,  &input_tensor, 1,
    output2_names, 1
  );

  // YOLO26 output shape: [1, 300, 6]
  auto & out2 = output2_tensors[0];
  auto shape2 = out2.GetTensorTypeAndShapeInfo().GetShape();
  const float * data2 = out2.GetTensorData<float>();
  size_t count2 = shape2[1] * shape2[2];  // 300 * 6

  output2 = std::vector<float>(data2, data2 + count2);

  auto t2 = std::chrono::high_resolution_clock::now();
  
  inference_time1 = std::chrono::duration<double, std::milli>(t1 - t0).count();
  inference_time2 = std::chrono::duration<double, std::milli>(t2 - t1).count();
}

cv::Mat YoloInference::preprocess(const cv::Mat & bgr_frame)
{
  // 1. Letterbox resize to 640x640 (preserve aspect ratio, pad with grey)
  int target = 640;
  float scale = std::min(
    static_cast<float>(target) / bgr_frame.cols,
    static_cast<float>(target) / bgr_frame.rows
  );
  int new_w = static_cast<int>(bgr_frame.cols * scale);
  int new_h = static_cast<int>(bgr_frame.rows * scale);
  int pad_x = (target - new_w) / 2;
  int pad_y = (target - new_h) / 2;

  cv::Mat resized;
  cv::resize(bgr_frame, resized, {new_w, new_h});

  cv::Mat letterboxed(target, target, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, new_w, new_h)));

  // 2. BGR → RGB, uint8 → float32, normalize to [0, 1]
  cv::Mat rgb;
  cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // 3. HWC → CHW (ONNX Runtime expects channel-first)
  // Split into 3 planes, then merge into a contiguous CHW blob
  std::vector<cv::Mat> channels(3);
  cv::split(rgb, channels);

  cv::Mat chw;
  cv::vconcat(channels, chw);  // stacks [H,W] planes vertically → [3H, W]
  return chw.reshape(1, 1);    // flatten to a row — pointer stays valid
}
