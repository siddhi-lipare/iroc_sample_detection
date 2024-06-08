import cv2
from ultralytics import YOLO
import numpy as np
import time
from typing import List, Tuple

# Load the model
model = YOLO("models/yolov8l-seg.pt")

# Define a video capture object
vid = cv2.VideoCapture(0)

# Define class IDs for bottle and cup (39 for bottle, 41 for cup in COCO dataset)
target_classes = [39, 41]

def is_cylinder(contour):
    if len(contour) < 5:
        return False
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        aspect_ratio = ma / (MA + 1e-6)
        if 0.4 < aspect_ratio < 0.6:  # Adjust the ratio based on cylinder properties
            return True
    return False

def get_sample_pixels(mask) -> List[Tuple[int, int]]:
    sample_pixels = []
    mask = mask.cpu().numpy().astype(np.uint8)
    mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    
    # Get coordinates of pixels that are part of the bottle
    pixels = np.argwhere(mask == 1)
    for pixel in pixels:
        sample_pixels.append((pixel[1], pixel[0]))  # (x, y) format

    return sample_pixels

# Initialize FPS calculation
prev_time = 0

# Create a file to save the sample pixels
file_name = "sample_pixels.txt"
file = open(file_name, "w")

while True:
    # Capture the video frame
    ret, frame = vid.read()

    if not ret:
        break

    # Calculate the current time and compute FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Make predictions
    results = model.predict(frame)[0]

    if results.masks is None or len(results.masks.data) == 0:
        print("No objects detected")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # Iterate through the segmented objects and draw outlines
    for idx, mask in enumerate(results.masks.data):
        class_id = int(results.boxes.cls[idx].item())

        # Only process if the detected object is a bottle or cup
        if class_id in target_classes:
            # Get sample pixels for the detected bottle or cup
            sample_pixels = get_sample_pixels(mask)

            # Write the sample pixels to the file
            file.write("Sample pixels for object {}: {}\n".format(idx, sample_pixels))

            # Mark the sample pixels on the frame
            for (x, y) in sample_pixels:
                frame[y, x] = (0, 0, 255)  # Mark the pixel in red

            # Convert the mask to a binary image
            mask = mask.cpu().numpy().astype(np.uint8)
            mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)[1]

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                color = (0, 255, 0) if is_cylinder(contour) else (0, 0, 255)

                # Create an RGB version of the mask
                mask_rgb = np.stack([mask * color[2], mask * color[1], mask * color[0]], axis=-1)

                # Blend the mask with the original frame
                frame = cv2.addWeighted(frame, 1, mask_rgb, 0.5, 0)

                # Draw contours on the frame
                cv2.drawContours(frame, [contour], -1, color, 2)

                # Get the class name
                class_name = results.names[class_id]

                # Find the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Put the class name near the segmented region
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # The 'q' button is set as the quitting button
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close the file
file.close()

# After the loop, release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
