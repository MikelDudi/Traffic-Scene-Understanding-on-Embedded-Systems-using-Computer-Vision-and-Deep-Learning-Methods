from ultralytics import YOLO
import cv2
import time
import os
import glob

# Load YOLO model1 only
model1 = YOLO(r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\test_scripts\best_320P_YOLOv8_ncnn_model")

# Define class names for model1
classes1 = ["traffic light", "speed limit", "crosswalk", "stop"]

# Directory containing images
image_dir = r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\val\images"  # <-- CHANGE this to your image folder
output_dir = r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\annotated_images"
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
image_paths = glob.glob(os.path.join(image_dir, "*.[jp][pn]g"))

# Process each image
for img_path in image_paths:
    print(f"\nProcessing image: {os.path.basename(img_path)}")
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"Failed to read image: {img_path}")
        continue

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
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Model 1 detections: {count1}")

    # Save annotated image
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved annotated image to: {output_path}")

    # Optional: display the image (press any key to continue to next)
    cv2.imshow("Detections", annotated_frame)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
