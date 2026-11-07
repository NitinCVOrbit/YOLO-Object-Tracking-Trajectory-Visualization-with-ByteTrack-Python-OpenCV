from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque
import random

# # ============================================================
# # Configuration
# # ============================================================

MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "01.mp4"

TRAJECTORY_LENGTH = 100
LINE_THICKNESS = 3
BOX_THICKNESS = 2
FONT_SCALE = 0.7

# ============================================================
# Load Model
# ============================================================

model = YOLO(MODEL_PATH)

# ============================================================
# Open Video
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Unable to open video.")

# ============================================================
# Dictionary for Trajectories
# ============================================================

# Create a dictionary to store the trajectory points of each tracked object.
trajectory_history = defaultdict(

    # Automatically create an empty deque when a new track ID appears.
    lambda: deque(

        # Store only the latest TRAJECTORY_LENGTH positions.
        maxlen=TRAJECTORY_LENGTH
    )
)

# # ============================================================
# # Dictionary for Colors
# # ============================================================

track_colors = {}

# # ============================================================
# # Random Color Generator
# # ============================================================


# Create a function to generate a color for each track ID.
def get_color(track_id):
    # Check whether this track ID already has a color.
    if track_id not in track_colors:
        # Generate a random BGR color and store it for this track ID.
        track_colors[track_id] = (random.randint(30, 255), random.randint(30, 255), random.randint(30, 255))
    # Return the color assigned to this track ID.
    return track_colors[track_id]

# ============================================================
# Main Loop
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # --------------------------------------------------------
    # Run YOLO + ByteTrack
    # --------------------------------------------------------

    # Run YOLO object detection and tracking on the current frame.
    results = model.track(
        frame,                    # Send the current video frame to YOLO.
        persist=True,             # Keep the same track IDs across video frames.
        tracker="bytetrack.yaml", # Use ByteTrack to track detected objects.
        verbose=False             # Do not print YOLO processing information.
    )

    # Get the first result from the results list.
    result = results[0]

    if result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # --------------------------------------------
            # Store trajectory
            # --------------------------------------------

            trajectory_history[track_id].append((cx, cy))

            color = get_color(track_id)

            # --------------------------------------------
            # Draw Bounding Box
            # --------------------------------------------

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)

            class_name = model.names[cls]
            label = f"{class_name} ID:{track_id}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                2
            )

            # --------------------------------------------
            # Draw Center Point
            # --------------------------------------------

            cv2.circle(frame, (cx, cy), 5, color, -1)

            # --------------------------------------------
            # Draw Trajectory
            # --------------------------------------------

            points = trajectory_history[track_id]
            # print(points)
            # break

            # Check whether we have at least two points to draw a line.A
            if len(points) > 1:
                # Start from the second point and go through all points.
                for i in range(1, len(points)):
                    # Draw a line between the previous point and the current point.
                    cv2.line(
                        frame,              # Draw the line on the current video frame.
                        points[i - 1],      # Starting point of the line.
                        points[i],          # Ending point of the line.
                        color,              # Color of the trajectory line.
                        LINE_THICKNESS      # Thickness of the trajectory line.
                    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow("Trajectory Tracking", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Finished!")

# Example: points = [(100, 200), (150, 250), (200, 300)]
# i = 1 → Connect (100, 200) to (150, 250).
# i = 2 → Connect (150, 250) to (200, 300).
# Start from index 1 because each line needs a previous point and a current point.S