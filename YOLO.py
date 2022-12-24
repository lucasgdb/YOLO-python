import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    _, frame = cap.read()

    result = model(frame)

    cv2.imshow('YOLO', np.squeeze(result.render()))

    if cv2.waitKey(10) & 0xFF == ord('c'):
        break

cap.release()
cv2.destroyAllWindows()
