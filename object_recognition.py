from ultralytics import YOLO
import cv2
import math

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the model
model = YOLO("yolo-Weights/yolov8n.pt")

# Teh classes that the model will recognise. The model is pretrained to all these classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Use an infinite loop to basically process each frame
while True:
    _, frame = cap.read()
    # Do the object recognition by sending the current frame used through the model.
    # The result is a list of objects recognised by the model as well as their coordinates
    results = model(frame, stream=True)

    for res in results:
        # Get all the boxes for the current result
        boxes = res.boxes

        for box in boxes:
            # Once we have the coordinates of a bounding box in the frame, it becomes straightforward to draw
            # rectangles on the live feed using opencv
            l, t, r, b = box.xyxy[0]
            l, t, r, b = int(l), int(t), int(r), int(b)  # convert to int values

            # Draw the bounding box in the webcam
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 255), 3)

            # This is just some descriptive stuff to put on the frame
            cls = int(box.cls[0])
            org = [l, t]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Object Recognition', frame)
    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('Object Recognition', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
