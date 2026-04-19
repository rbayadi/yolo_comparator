#include "yolo26_post_processing.hpp"

std::vector<Detection> postprocess_yolo26_output(
    const std::vector<float>& yolo_output,
    int img_width,
    int img_height,
    float conf_threshold
) {
    const int elements_per_box = 6;
    const int num_boxes = 300;

    // Letterbox scale/padding
    const int target = 640;
    float scale = std::min(
        static_cast<float>(target) / img_width,
        static_cast<float>(target) / img_height
    );
    int new_w = static_cast<int>(img_width * scale);
    int new_h = static_cast<int>(img_height * scale);
    int pad_x = (target - new_w) / 2;
    int pad_y = (target - new_h) / 2;

    std::vector<Detection> detections;

    for (int i = 0; i < num_boxes; ++i) {
        const float* data = &yolo_output[i * elements_per_box];

        float conf     = data[4];
        if (conf < conf_threshold)
            continue;

        int class_id   = static_cast<int>(data[5]);

        float x1 = data[0];
        float y1 = data[1];
        float x2 = data[2];
        float y2 = data[3];

        // Reverse letterbox
        float x1_orig = (x1 - pad_x) / scale;
        float y1_orig = (y1 - pad_y) / scale;
        float x2_orig = (x2 - pad_x) / scale;
        float y2_orig = (y2 - pad_y) / scale;

        x1_orig = std::max(0.0f, x1_orig);
        y1_orig = std::max(0.0f, y1_orig);
        x2_orig = std::min(static_cast<float>(img_width),  x2_orig);
        y2_orig = std::min(static_cast<float>(img_height), y2_orig);

        Detection det;
        det.box      = cv::Rect(
            static_cast<int>(x1_orig),
            static_cast<int>(y1_orig),
            static_cast<int>(x2_orig - x1_orig),
            static_cast<int>(y2_orig - y1_orig)
        );
        det.score    = conf;
        det.class_id = class_id;
        detections.push_back(det);
    }

    return detections;
}