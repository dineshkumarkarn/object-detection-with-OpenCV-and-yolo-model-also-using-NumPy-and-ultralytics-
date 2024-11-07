from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np

np.random.seed(20)

#load model
model = YOLO('trainedyolov8.pt')

#capture videos from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit(0)

while True:
    ret, img = cap.read()

    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img,conf=0.2)

    for r in results:

        annotator = Annotator(img)

        boxes = r.boxes

        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)], color=(255, 255,0))


    img = annotator.result()
    cv2.imshow('YOLO V8 Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done release the capture
cap.release()
cv2.destroyAllWindows()
