# computer-vision

## Project: OpenCV_ObjectDetection_YOLO_Pics

Goal: Detect objects on pictures

Description: OpenCV implementation for object detection on images

Model: YOLOv3 trained on COCO data set by Darknet in 2018

Code source: Code based on work by Adrian Reosebrock (www.pyimagesearch.com)

Python program file: yolo_pic.py

The program is executed by command line arguments:

```bash
python yolo_pic.py --image images/traffic_jam.jpg --yolo yolo-coco

python yolo_pic.py --image images/airport_field.jpg --yolo yolo-coco

python yolo_pic.py --image images/bike_car.jpg --yolo yolo-coco --confidence 0.3

python yolo_pic.py --image images/agr_field_1.jpg --yolo yolo-coco
```
