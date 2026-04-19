<b> Compares detection performance of YOLOV5 and YOLO26 within a ROS2 node </b>

<img width="1858" height="956" alt="image" src="https://github.com/user-attachments/assets/f7acd8f1-6db9-424e-840d-16aa75fa6f90" />


Runs on Ubuntu 24.2 (may be earlier ones as well). Prerequisites:
*  ROS2 Kilted Kaiju
*  ONNX run time: Download from https://github.com/microsoft/onnxruntime/releases/download/v1.24.4/onnxruntime-linux-x64-1.24.4.tgz. Extract and move the folder to /opt and rename to onnxruntime
*  FoxGlove

How to build:

```bash
./build_node
```
How to run:
```bash
./run_node_bridge
```
How to stop:
```bash
./stop_node_bridge
```
Comparison KPIs based on nuScenes mini data:
*  Localization accuracy or bounding box tightness: YOLO26 has a small edge
*  Detection at far range: YOLOV5 has a small edge
*  Detection stability: YOLOV5 has a small edge
*  Class stability: YOLOV5 has a small edge
*  Precision: YOLO26 does much better in false positives
*  Inference time (includes NMS in case of YOLOV5): YOLOV5 is the clear winner
