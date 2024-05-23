import cv2
import numpy as np
import urllib.request
import os

# URLs for the YOLO model files
cfg_url = 'https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg'
weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
names_url = 'https://github.com/pjreddie/darknet/raw/master/data/coco.names'

# Filenames for the saved files
cfg_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'
names_file = 'coco.names'

# Function to download a file
def download_file(url, filename):
    if not os.path.isfile(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"{filename} downloaded")
    else:
        print(f"{filename} already exists")

# Download the files
download_file(cfg_url, cfg_file)
download_file(weights_url, weights_file)
download_file(names_url, names_file)

# Load YOLO
net = cv2.dnn.readNet(weights_file, cfg_file)

# Enable CUDA if available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class labels
with open(names_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Classes loaded:", classes)

def detect_objects(frame):
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)
            print(f"Detected {label} with confidence {confidences[i]} at ({x}, {y}, {w}, {h})")
    else:
        print("No objects detected")

    return frame

# Ask the user for input
option = input("Enter '1' to detect objects in an image or '2' to use the webcam: ")

if option == '1':
    # Detect objects in an image
    image_path = input("Enter the path to the image: ")
    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        result = detect_objects(image)
        cv2.imshow("Image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"No such file: {image_path}")

elif option == '2':
    # Use the webcam
    cap = cv2.VideoCapture(0)
    frame_skip = 2  # Skip frames to reduce load
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_id += 1
        if frame_id % frame_skip != 0:
            continue

        result = detect_objects(frame)
        cv2.imshow("Webcam", result)

        # Stop the script when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid option. Please enter '1' or '2'.")
