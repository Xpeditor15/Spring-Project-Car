import cv2
import numpy as np
import os
from picamera2 import Picamera2
from collections import deque
import math
import uuid

# Initialize Raspberry Pi Camera
def initialize_camera():
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
        picam2.start()
        return picam2
    except RuntimeError as e:
        print(f"Camera initialization failed: {e}")
        return None

picam2 = initialize_camera()
if picam2 is None:
    print("Exiting program. Camera could not be initialized.")
    exit()

# Color thresholds for shape detection (HSV ranges)
COLOR_RANGES = {
    'red': ([0, 100, 50], [10, 255, 255]),
    'green': ([35, 100, 100], [85, 255, 255]),
    'blue': ([95, 50, 50], [145, 255, 255]),
    'yellow': ([15, 100, 100], [45, 255, 255])
}

# Blue range for arrow detection
ARROW_BLUE_RANGE = ([95, 50, 50], [145, 255, 255])  # Matches blue in COLOR_RANGES

def preprocess_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for color, (lower, upper) in COLOR_RANGES.items():
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.bitwise_or(mask, color_mask)
    track_mask = cv2.inRange(hsv, np.array([30, 50, 50]), np.array([50, 100, 100]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(track_mask))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def create_debug_frame(frame, mask, shapes, arrow_dir, blue_mask, debug_contours):
    # Create a blank debug frame (2x2 grid for visualization)
    debug_height, debug_width = frame.shape[0], frame.shape[1]
    debug_frame = np.zeros((debug_height * 2, debug_width * 2, 3), dtype=np.uint8)
    
    # Top-left: Original frame with shapes and arrow annotations
    annotated_frame = frame.copy()
    for x, y, w, h, label in shapes:
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if arrow_dir:
        cv2.putText(annotated_frame, f"Arrow: {arrow_dir}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    debug_frame[0:debug_height, 0:debug_width] = annotated_frame

    # Top-right: Preprocessed mask for shape detection
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_rgb, "Shape Detection Mask", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    debug_frame[0:debug_height, debug_width:debug_width*2] = mask_rgb

    # Bottom-left: Blue mask for arrow detection
    blue_mask_rgb = cv2.cvtColor(blue_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(blue_mask_rgb, "Arrow Blue Mask", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    debug_frame[debug_height:debug_height*2, 0:debug_width] = blue_mask_rgb

    # Bottom-right: Contour debug frame
    contour_frame = debug_contours.copy()
    cv2.putText(contour_frame, "Contours with Annotations", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    debug_frame[debug_height:debug_height*2, debug_width:debug_width*2] = contour_frame

    return debug_frame

def detect_colored_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = preprocess_frame(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    max_area = 0.5 * frame.shape[0] * frame.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500 or area > max_area:
            continue
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)
        roi = hsv[y:y+h, x:x+w]
        for color, (lower, upper) in COLOR_RANGES.items():
            color_mask = cv2.inRange(roi, np.array(lower), np.array(upper))
            if cv2.countNonZero(color_mask) > 0.1 * w * h:
                break
        else:
            color = "unknown"
        sides = len(approx)
        print(f"Shape Contour: Sides = {sides}, Area = {area}")
        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            aspect_ratio = float(w)/h
            shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
        elif sides == 5:
            shape = "Pentagon"
        elif sides == 6:
            shape = "Hexagon"
        else:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            print(f"Shape Circularity: {circularity}")
            shape = "Circle" if circularity > 0.5 else "Unknown"
        shapes.append((x, y, w, h, f"{color} {shape}"))
    return shapes, mask

def detect_arrow_direction(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue, upper_blue = np.array(ARROW_BLUE_RANGE[0]), np.array(ARROW_BLUE_RANGE[1])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((5,5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    arrow_direction = None
    max_area = 0.5 * frame.shape[0] * frame.shape[1]
    min_area = 500
    valid_contour = None
    largest_area = 0

    debug_frame = frame.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            print(f"Arrow Contour: Area = {area}, Skipped (area out of range)")
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        if area > largest_area:
            largest_area = area
            valid_contour = contour

    if valid_contour is None:
        print("No valid arrow contours found")
        return None, blue_mask, debug_frame

    epsilon = 0.02 * cv2.arcLength(valid_contour, True)
    approx = cv2.approxPolyDP(valid_contour, epsilon, True)

    area = cv2.contourArea(valid_contour)
    perimeter = cv2.arcLength(valid_contour, True)
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h

    is_rectangle = (circularity < 0.6 and (aspect_ratio > 1.2 or aspect_ratio < 0.8))
    is_circle = (circularity > 0.6 or len(approx) > 6)

    if not (is_rectangle or is_circle):
        print(f"Arrow Contour: Area = {area}, Sides = {len(approx)}, Circularity = {circularity}, Aspect = {aspect_ratio}, Skipped (not rectangle or circle)")
        return None, blue_mask, debug_frame

    M = cv2.moments(valid_contour)
    if M["m00"] == 0:
        print(f"Arrow Contour: Area = {area}, Skipped (invalid moments)")
        return None, blue_mask, debug_frame
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    max_dist = 0
    tip_point = None
    for point in valid_contour:
        x, y = point[0]
        dist = math.sqrt((x - cx)**2 + (y - cy)**2)
        if dist > max_dist:
            max_dist = dist
            tip_point = (x, y)

    if tip_point is None:
        print(f"Arrow Contour: Area = {area}, Skipped (no tip found)")
        return None, blue_mask, debug_frame

    shape_type = "Rectangle" if is_rectangle else "Circle"
    if is_rectangle:
        dx = tip_point[0] - cx
        arrow_direction = "right" if dx > 0 else "left"
    elif is_circle:
        dy = tip_point[1] - cy
        arrow_direction = "up" if dy < 0 else "down"

    cv2.circle(debug_frame, (cx, cy), 5, (255, 0, 0), -1)
    cv2.circle(debug_frame, tip_point, 5, (0, 0, 255), -1)
    cv2.line(debug_frame, (cx, cy), tip_point, (255, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(valid_contour)
    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    cv2.putText(debug_frame, f"{shape_type}: {arrow_direction}", (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    print(f"Arrow Contour: Area = {area}, Sides = {len(approx)}, Circularity = {circularity}, Aspect = {aspect_ratio}, Shape = {shape_type}, Direction = {arrow_direction}, Tip = {tip_point}")

    return arrow_direction, blue_mask, debug_frame

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(param, cv2.COLOR_BGR2HSV)
        print(f"HSV at ({x}, {y}): {hsv[y, x]}")

def main_loop():
    prev_detections = deque(maxlen=5)
    cv2.namedWindow("Track Symbol Detection")
    cv2.namedWindow("Debug Visualization")
    while True:
        frame = picam2.capture_array()
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        cv2.setMouseCallback("Track Symbol Detection", mouse_callback, frame)
        cv2.imwrite("raw_frame.jpg", frame)
        shapes, mask = detect_colored_shapes(frame)
        arrow_dir, blue_mask, debug_contours = detect_arrow_direction(frame)
        debug_frame = create_debug_frame(frame, mask, shapes, arrow_dir, blue_mask, debug_contours)
        cv2.imshow("Track Symbol Detection", frame)
        cv2.imshow("Debug Visualization", debug_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    picam2.stop()

if __name__ == "__main__":
    main_loop()