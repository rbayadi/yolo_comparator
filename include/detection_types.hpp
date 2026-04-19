#pragma once
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};
