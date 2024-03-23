import os
import numpy
import cv2
from ultralytics import YOLO

def preprocess(image):
    # image = cv2.imread(image)
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        pass
    image = cv2.resize(image, (256, 256))
    image = numpy.array(image)
    return image

def predictyolo(image):
    model = YOLO('best.pt')
    results = model([image])
    detected = False
    # print(results)
    for result in results:
        boxes = result.boxes
        if len(boxes)>0:
            detected = True
        result.save(filename='result.jpg')
        return detected

        