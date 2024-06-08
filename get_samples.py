import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

# Load the model
model = YOLO("models/yolov8l-seg.pt")

# Define class IDs for bottle (39 in COCO dataset)
bottle_class_id = 39

def get_sample_pixels(img) -> List[Tuple[int, int]]:
    results = model.predict(img)[0]
    sample_pixels = []

    if results.masks is None or len(results.masks.data) == 0:
        return sample_pixels  # Return empty list if no objects detected

    for idx, mask in enumerate(results.masks.data):
        class_id = int(results.boxes.cls[idx].item())

        if class_id == bottle_class_id:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

            # Get coordinates of pixels that are part of the bottle
            pixels = np.argwhere(mask == 1)
            for pixel in pixels:
                sample_pixels.append((pixel[1], pixel[0]))  # (x, y) format

    return sample_pixels

def main():
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Detect and process frame
        sample_pixels = get_sample_pixels(frame)

        # Print coordinates of sample pixels
        print("Sample Pixels Coordinates:")
        for pixel in sample_pixels:
            print(pixel)

        # Draw sample pixels on the frame
        for (x, y) in sample_pixels:
            frame[y, x] = (0, 0, 255)  # Mark the pixel in red

        # Display the resulting frame
        cv2.imshow("YOLOv8 Cylinder Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
