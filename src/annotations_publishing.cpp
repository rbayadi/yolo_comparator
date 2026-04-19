#include "annotations_publishing.hpp"

const std::vector<std::string> coco_labels = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
};

void publish_annotations(
    const rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>::SharedPtr publisher,
    const std::vector<Detection>& detections,
    const foxglove_msgs::msg::Color annotations_color,
    const std_msgs::msg::Header& header)
{
    foxglove_msgs::msg::ImageAnnotations annotations_msg;
    annotations_msg.circles.clear();
    annotations_msg.points.clear();

    for (const auto & det : detections) {
        // det assumed to have: bbox center_x, center_y, size_x, size_y
        foxglove_msgs::msg::PointsAnnotation rect;
        rect.timestamp = header.stamp;
        rect.type = foxglove_msgs::msg::PointsAnnotation::LINE_LOOP;  // closed rectangle
        rect.thickness = 2.0;

        // fill RGBA color — green
        rect.outline_color = annotations_color;

        // Bounding box
        float bbox_center_x = det.box.x + det.box.width / 2.0;
        float bbox_center_y = det.box.y + det.box.height / 2.0;

        float bbox_size_x = det.box.width;
        float bbox_size_y = det.box.height;

        // compute corners from center + size
        float x0 = bbox_center_x - bbox_size_x / 2.0;
        float y0 = bbox_center_y - bbox_size_y / 2.0;
        float x1 = bbox_center_x + bbox_size_x / 2.0;
        float y1 = bbox_center_y + bbox_size_y / 2.0;

        // four corners in pixel coords
        foxglove_msgs::msg::Point2 tl, tr, br, bl;
        tl.x = x0; tl.y = y0;
        tr.x = x1; tr.y = y0;
        br.x = x1; br.y = y1;
        bl.x = x0; bl.y = y1;

        rect.points = {tl, tr, br, bl};
        annotations_msg.points.push_back(rect);

        // Text labels
        foxglove_msgs::msg::TextAnnotation text;
        text.timestamp = header.stamp;

        text.text = coco_labels[det.class_id];  // or label

        text.position.x = x0;
        text.position.y = y1;

        text.text_color = annotations_color;

        text.font_size = 20.0;

        annotations_msg.texts.push_back(text);        
    }

    publisher->publish(annotations_msg);   
}
