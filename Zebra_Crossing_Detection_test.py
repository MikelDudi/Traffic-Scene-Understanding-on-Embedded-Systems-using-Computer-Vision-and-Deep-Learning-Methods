import cv2
import os
import glob

# === CONFIGURATION ===
ROI_TOP = 0.7    # Top % of the frame to start ROI
ROI_BOTTOM = 0.9 # Bottom % of the frame to end ROI
ROI_LEFT = 0.0   # Left margin (% of width)
ROI_RIGHT = 1.0  # Right margin (% of width)

# Path to folder with images
image_dir = "/home/mikeldudi/Desktop/pattern_recognition_project/test_images/zebra_crossing_test"  # <-- change this
image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
              glob.glob(os.path.join(image_dir, "*.jpeg")) + \
              glob.glob(os.path.join(image_dir, "*.png"))


for img_path in image_paths:
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load {img_path}")
        continue

    H, W = frame.shape[:2]

    # Define ROI bounds
    y0 = int(ROI_TOP * H)
    y1 = int(ROI_BOTTOM * H)
    x0 = int(ROI_LEFT * W)
    x1 = int(ROI_RIGHT * W)

    # Extract ROI for processing
    roi = frame[y0:y1, x0:x1]

    # Convert to HSV and mask bright white areas
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 25, 255))  # stricter white

    # Morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    morph = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contour detection
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h + 1e-5)
        if w > 60 and h > 10 and 2.5 < aspect < 8:
            stripe_boxes.append((x, y, w, h))

    # Group horizontal-aligned stripes
    stripe_boxes = sorted(stripe_boxes, key=lambda b: b[1])
    groups, group = [], [stripe_boxes[0]] if stripe_boxes else []
    for i in range(1, len(stripe_boxes)):
        if abs(stripe_boxes[i][1] - stripe_boxes[i-1][1]) < 50:
            group.append(stripe_boxes[i])
        else:
            if len(group) >= 3:
                groups.append(group)
            group = [stripe_boxes[i]]
    if len(group) >= 3:
        groups.append(group)

    # Draw detected zebra crossing boxes on the full frame
    detected = False
    for g in groups:
        gx = min(b[0] for b in g)
        gy = min(b[1] for b in g)
        gw = max(b[0] + b[2] for b in g) - gx
        gh = max(b[1] + b[3] for b in g) - gy
        cv2.rectangle(frame, (gx + x0, gy + y0), (gx + gw + x0, gy + gh + y0), (0, 255, 255), 3)
        detected = True

    # Show ROI bounds on full frame
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), -1)
    alpha = 0.15  # Transparency level
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Detection status text
    txt = "ZEBRA CROSSING: YES" if detected else "ZEBRA CROSSING: NO"
    color = (0, 255, 0) if detected else (0, 0, 255)
    cv2.putText(frame, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Display results
    cv2.imshow("Zebra Crossing Detection", frame)
    #cv2.imshow("Mask", mask)
    #cv2.imshow("Morph", morph)

    print(f"{os.path.basename(img_path)} - Detected: {detected}")

    key = cv2.waitKey(0)  # Wait for key press to move to next image
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
