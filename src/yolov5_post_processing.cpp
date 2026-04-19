#include "yolov5_post_processing.hpp"

std::vector<Detection> postprocess_yolov5_output(
    const std::vector<float>& yolo_output,
    int img_width,
    int img_height,
    float conf_threshold,
    float nms_threshold
) {
    const int elements_per_box = 85;
    const int num_boxes = yolo_output.size() / elements_per_box;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;

    // Calculate the scale and padding used during preprocessing
    // (must match the letterbox transformation used in preprocessing)
    const int target = 640;
    float scale = std::min(
        static_cast<float>(target) / img_width,
        static_cast<float>(target) / img_height
    );
    int new_w = static_cast<int>(img_width * scale);
    int new_h = static_cast<int>(img_height * scale);
    int pad_x = (target - new_w) / 2;
    int pad_y = (target - new_h) / 2;

    for (int i = 0; i < num_boxes; ++i) {
        const float* data = &yolo_output[i * elements_per_box];

        float cx = data[0];
        float cy = data[1];
        float w  = data[2];
        float h  = data[3];
        float obj_conf = data[4];

        // Find best class
        float max_class_score = 0.0f;
        int class_id = -1;

        for (int c = 5; c < elements_per_box; ++c) {
            if (data[c] > max_class_score) {
                max_class_score = data[c];
                class_id = c - 5;
            }
        }

        float confidence = obj_conf * max_class_score;

        if (confidence < conf_threshold)
            continue;

        // Convert to corner format (still in 640x640 letterboxed space)
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // Reverse the letterbox transformation:
        // 1. Remove padding offset
        float x1_unpadded = x1 - pad_x;
        float y1_unpadded = y1 - pad_y;
        float x2_unpadded = x2 - pad_x;
        float y2_unpadded = y2 - pad_y;

        // 2. Scale back to original image dimensions
        float x1_orig = x1_unpadded / scale;
        float y1_orig = y1_unpadded / scale;
        float x2_orig = x2_unpadded / scale;
        float y2_orig = y2_unpadded / scale;

        // Clamp to image boundaries
        x1_orig = std::max(0.0f, x1_orig);
        y1_orig = std::max(0.0f, y1_orig);
        x2_orig = std::min(static_cast<float>(img_width), x2_orig);
        y2_orig = std::min(static_cast<float>(img_height), y2_orig);

        int left   = static_cast<int>(x1_orig);
        int top    = static_cast<int>(y1_orig);
        int width  = static_cast<int>(x2_orig - x1_orig);
        int height = static_cast<int>(y2_orig - y1_orig);

        boxes.emplace_back(left, top, width, height);
        scores.push_back(confidence);
        class_ids.push_back(class_id);
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.box = boxes[idx];
        det.score = scores[idx];
        det.class_id = class_ids[idx];
        detections.push_back(det);
    }

    return detections;
}
