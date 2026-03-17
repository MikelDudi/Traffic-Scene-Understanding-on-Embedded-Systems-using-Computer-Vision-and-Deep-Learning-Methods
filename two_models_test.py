import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
VIDEO_SOURCE = "/home/mikeldudi/Downloads/Η ταινία μου 1.mp4"  # 0 for webcam
FRAME_W = 640
ROI_TOP = 0.7
ROI_BOTTOM = 0.95

# =========================
# ONNX YOLO CONFIG
# =========================
model_path = "/home/mikeldudi/Desktop/pattern_recognition_project/best.onnx"
net = cv2.dnn.readNetFromONNX(model_path)
classes = ["traffic light", "speed limit", "crosswalk", "stop"]
input_size = 320
conf_threshold = 0.6
nms_threshold = 0.5

# =========================
# NCNN YOLO CONFIG
# =========================
ncnn_model = YOLO("yolov8n_ncnn_model", task="detect")  # You confirmed this works

# =========================
# Frame skipping + cache
# =========================
skip_interval = 1
frame_count = 0
cached_boxes, cached_confidences, cached_class_ids = [], [], []

# =========================
# HELPERS
# =========================
def resize_keep_ar(frame, target_w):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / float(w)
    return cv2.resize(frame, (target_w, int(round(h * scale))), interpolation=cv2.INTER_AREA)

# =========================
# MAIN
# =========================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("Could not open video source")

trail = deque(maxlen=16)
fps = 0
prev_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = resize_keep_ar(frame, FRAME_W)
    H, W = frame.shape[:2]

    # =========================
    # YOLO ONNX INFERENCE (OpenCV)
    # =========================
    if frame_count % skip_interval == 0:
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        boxes, confidences, class_ids = [], [], []
        for det in outputs[0]:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * det[4]
            if confidence > conf_threshold:
                cx, cy, w, h = det[0:4]
                x = int((cx - w/2) * W / input_size)
                y = int((cy - h/2) * H / input_size)
                w = int(w * W / input_size)
                h = int(h * H / input_size)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        cached_boxes = [boxes[i[0] if isinstance(i, (list, tuple, np.ndarray)) else i] for i in indices]
        cached_confidences = [confidences[i[0] if isinstance(i, (list, tuple, np.ndarray)) else i] for i in indices]
        cached_class_ids = [class_ids[i[0] if isinstance(i, (list, tuple, np.ndarray)) else i] for i in indices]

    # =========================
    # DRAW ONNX YOLO RESULTS
    # =========================
    for box, conf, class_id in zip(cached_boxes, cached_confidences, cached_class_ids):
        x, y, w, h = box
        label = f"[F] {classes[class_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # =========================
    # NCNN YOLO INFERENCE
    # =========================
    ncnn_results = ncnn_model.predict(frame, imgsz=320, verbose=False)[0]

    for box in ncnn_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        label = f"[P] {ncnn_model.names[class_id]}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # =========================
    # ZEBRA CROSSING DETECTION
    # =========================
    y0 = int(ROI_TOP * H)
    y1 = int(ROI_BOTTOM * H)
    roi = frame[y0:y1, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 150), (180, 30, 255))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h + 1e-5)
        if w > 40 and 2 < aspect < 15:
            stripe_boxes.append((x, y, w, h))

    stripe_boxes = sorted(stripe_boxes, key=lambda b: b[1])
    groups, group = [], [stripe_boxes[0]] if stripe_boxes else []
    for i in range(1, len(stripe_boxes)):
        if abs(stripe_boxes[i][1] - stripe_boxes[i-1][1]) < 60:
            group.append(stripe_boxes[i])
        else:
            if len(group) >= 3:
                groups.append(group)
            group = [stripe_boxes[i]]
    if len(group) >= 3:
        groups.append(group)

    detected = False
    for g in groups:
        gx = min(b[0] for b in g)
        gy = min(b[1] for b in g)
        gw = max(b[0] + b[2] for b in g) - gx
        gh = max(b[1] + b[3] for b in g) - gy
        cv2.rectangle(frame, (gx, gy + y0), (gx + gw, gy + gh + y0), (0, 255, 255), 3)
        detected = True

    txt = "ZEBRA CROSSING: YES" if detected else "ZEBRA CROSSING: NO"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # =========================
    # FPS DISPLAY
    # =========================
    curr_time = cv2.getTickCount()
    fps = cv2.getTickFrequency() / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # =========================
    # DISPLAY
    # =========================
    cv2.imshow("YOLO Models + Zebra Crossing", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
