from ultralytics import YOLO
import cv2
import time

# Load only model2
model2 = YOLO(r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\test_scripts\pedestrian_yolov8n_ncnn_model")

# Define ROI area (as % of image width/height)
ROI_TOP = 0.05
ROI_BOTTOM = 0.9
ROI_LEFT = 0.1
ROI_RIGHT = 0.8

# Class ID for 'person' in YOLOv8 (default is 0)
PERSON_CLASS_ID = 0

#video_path =0
video_path = r"C:\Users\arsen\OneDrive\Desktop\Pattern Recognition project\images and videos\Η ταινία μου 1.mp4"
# Open webcam (or use video file)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # Calculate ROI coordinates
    x0 = int(ROI_LEFT * W)
    x1 = int(ROI_RIGHT * W)
    y0 = int(ROI_TOP * H)
    y1 = int(ROI_BOTTOM * H)

    start_time = time.time()

    # Run model2 inference
    results2 = model2.predict(source=frame, conf=0.5, iou=0.5, verbose=False)

    annotated_frame = frame.copy()
    danger = False  # Flag for detection in ROI

    # Draw all detections
    boxes2 = results2[0].boxes
    for box in boxes2:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
        label = f"id{cls_id}: {conf:.2f}"
        color = (255, 0, 0)

        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1_box, y1_box), (x2_box, y2_box), color, 2)
        cv2.putText(annotated_frame, label, (x1_box, y1_box - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # === Check for PERSON in ROI ===
        if cls_id == PERSON_CLASS_ID:
            # Check if center of person box is inside ROI
            center_x = (x1_box + x2_box) // 2
            center_y = (y1_box + y2_box) // 2
            if x0 <= center_x <= x1 and y0 <= center_y <= y1:
                danger = True

    # Draw ROI box
    cv2.rectangle(annotated_frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

    # Display danger message if person detected in ROI
    if danger:
        warning_text = "🚨 DANGER: Person in path! BRAKE! 🚨"
        cv2.putText(annotated_frame, warning_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Show FPS
    fps = 1.0 / (time.time() - start_time + 1e-6)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show annotated frame
    cv2.imshow("Driver Assist - Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
