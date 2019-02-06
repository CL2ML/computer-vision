# OpenCV Object Detection - YOLOv3 in python

"""
General information: The Code is based on work by Adrian Rosebrock. 

More info can be found here: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

Used model: YOLOv3 (2018) trained on the COCO dataset

The COCO dataset consists of 80 labels, including, but not limited to:

    People
    Bicycles
    Cars and trucks
    Airplanes
    Stop signs and fire hydrants
    Animals, including cats, dogs, birds, horses, cows, and sheep, to name a few
    Kitchen and dining objects, such as wine glasses, cups, forks, knives, spoons, etc.

Data sources:

Darknet Git Repository: https://github.com/pjreddie/darknet
--> Training labels: coco.names 
--> Model specs: yolov3.cfg

yolov3.weights: https://pjreddie.com/darknet/yolo/ 
"""

  
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
 

# Construct the argument parse for the command line and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
# Create dictionary containing the key-value pairs for the command line arguments
args = vars(ap.parse_args())

# CLASS LABELS
#-------------
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
 
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# WEIGHTS 
#-------- 
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
 
# LOAD DNN MODEL
#--------------- 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# LOAD IMAGE
#-----------
# load our input image and grab its spatial dimensions
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image
# then perform a forward pass of the YOLO object detector
# giving us our bounding boxes and associated probabilities
# The pixel values are normalized
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
net.setInput(blob)
# Send image through the network and track running time
start = time.time()     
layerOutputs = net.forward(ln)
end = time.time()

# show timing information on YOLO
print("[INFO] YOLO took {:.6f} seconds".format(end - start))

# VISUALIZE RESULTS
#------------------
# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
boxes = []
confidences = []
classIDs = []

# Fill lists with the model data:
# loop over each of the layer outputs
for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)


# SUPPRESSING OVERLAPPING BOUNDING BOXES
#---------------------------------------
# apply non-maxima suppression to suppress weak, overlapping bounding boxes
# YOLO does not apply non-maxima suppression for us, so we need to explicitly apply it
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])


# DRAW BOXES AND CLASS TEXT ON IMAGE
#----------------------------------
# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)


# SHOW OUTPUT IMAGE
#------------------
cv2.imshow("Image", image)
cv2.waitKey(0)  # Wait until key is pressed

