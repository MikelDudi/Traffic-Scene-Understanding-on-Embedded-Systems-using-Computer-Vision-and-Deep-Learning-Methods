from ultralytics import YOLO
import cv2
import time
import os

# Load YOLO model1 only
model1 = YOLO(r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\test_scripts\best_320P_YOLOv8_ncnn_model")

# Define class names for model1
classes1 = ["traffic light", "speed limit", "crosswalk", "stop"]

# --- VIDEO/WEBCAM INPUT ---
# To use a video file, set video_path to your file path, e.g.,
# video_path = "/home/mikeldudi/Desktop/pattern_recognition_project/test_video.mp4"
# To use webcam, set video_path = 0
video_path = 0  # <-- 0 for webcam, or path to video file
video_path = r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\images and videos\Η ταινία μου 1.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

# Optional: output video
output_dir = "/home/mikeldudi/Desktop/pattern_recognition_project/annotated_videos"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output.avi")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

# Define codec and create VideoWriter object
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or failed to read frame.")
        break

    start_time = time.time()

    # Run inference on model1 only
    results1 = model1.predict(source=frame, conf=0.5, iou=0.5, verbose=False)

    inference_time = time.time() - start_time

    annotated_frame = frame.copy()

    # Draw results from model 1
    boxes1 = results1[0].boxes
    count1 = len(boxes1)

    for box in boxes1:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{classes1[cls_id]}: {conf:.2f}" if cls_id < len(classes1) else f"id{cls_id}: {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Print info
    print(f"Inference time: {inference_time:.3f} seconds | Model 1 detections: {count1}")

    # Write frame to output video
    out.write(annotated_frame)

    # Display the frame
    cv2.imshow("Detections", annotated_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

# --- IMAGE FOLDER PROCESSING (COMMENTED OUT) ---
"""
import glob

# Directory containing images
image_dir = "/home/mikeldudi/Desktop/pattern_recognition_project/test_images/val/images"
os.makedirs(output_dir, exist_ok=True)
image_paths = glob.glob(os.path.join(image_dir, "*.[jp][pn]g"))

for img_path in image_paths:
    print(f"\nProcessing image: {os.path.basename(img_path)}")
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Failed to read image: {img_path}")
        continue

    start_time = time.time()
    results1 = model1.predict(source=frame, conf=0.5, iou=0.5, verbose=False)
    inference_time = time.time() - start_time

    annotated_frame = frame.copy()
    boxes1 = results1[0].boxes
    count1 = len(boxes1)

    for box in boxes1:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{classes1[cls_id]}: {conf:.2f}" if cls_id < len(classes1) else f"id{cls_id}: {conf:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print(f"Inference time: {inference_time:.3f} seconds | Model 1 detections: {count1}")
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated image to: {output_path}")
    cv2.imshow("Detections", annotated_frame)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
"""
