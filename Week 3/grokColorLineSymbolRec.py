import RPi.GPIO as GPIO
import time
import math
import numpy as np
import cv2
from picamera2 import Picamera2

# --------------------------------------------------------------------------------
#                               GPIO / MOTOR SETUP
# --------------------------------------------------------------------------------

# Define GPIO pins
IN1, IN2 = 23, 24         # Left motor control
IN3, IN4 = 27, 22         # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 18      # Right encoder
encoderPinLeft = 19       # Left encoder
ServoMotor = 26           # Servo motor PWM for the camera

# Constants for line following
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5      # Duty cycle for 180 degrees
SERVO_FREQ = 50            # 50Hz frequency for servo

# Line following parameters
BASE_SPEED = 50            # Base motor speed (0-100)
TURN_SPEED = 65            # Speed for pivot turns (0-100)
MIN_CONTOUR_AREA = 250     # Minimum area for valid contours
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Color detection ranges
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 120])
LOWER_RED1 = np.array([0, 120, 70])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_BLUE = np.array([95, 20, 150])
UPPER_BLUE = np.array([115, 255, 255])
LOWER_GREEN = np.array([65, 20, 150])
UPPER_GREEN = np.array([85, 100, 255])
LOWER_YELLOW = np.array([85, 20, 150])
UPPER_YELLOW = np.array([105, 100, 255])

TURN_THRESHOLD = 60         # Error threshold for pivoting
REVERSE_DURATION = 0.3      # Seconds to reverse
REVERSE_SPEED = 40          # Speed when reversing

# Scanning angles for servo
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5

# Global counters
right_counter = 0
left_counter = 0

# Encoder callbacks
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    right_pwm = GPIO.PWM(ENA, 1000)
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

def set_servo_angle_simple(servo_pwm, angle):
    angle = max(0, min(180, angle))
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servo_pwm.ChangeDutyCycle(0)

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def move_forward(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(BASE_SPEED)
    left_pwm.ChangeDutyCycle(BASE_SPEED)

def move_backward(right_pwm, left_pwm, speed):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    right_pwm.ChangeDutyCycle(speed)
    left_pwm.ChangeDutyCycle(speed)

def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    turn_time = abs(scanned_angle - 90) / 45.0
    if scanned_angle > 90:
        print(f"Detected angle {scanned_angle}: pivoting LEFT for {turn_time:.2f} seconds")
        pivot_turn_left(right_pwm, left_pwm)
    elif scanned_angle < 90:
        print(f"Detected angle {scanned_angle}: pivoting RIGHT for {turn_time:.2f} seconds")
        pivot_turn_right(right_pwm, left_pwm)
    else:
        print("Servo at 90°: no pivot required.")
        return
    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)
    print("Resetting servo to 90 degrees")
    set_servo_angle_simple(servo_pwm, 90)

# --------------------------------------------------------------------------------
#                           LINE DETECTION
# --------------------------------------------------------------------------------

def detect_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    center_x_frame = FRAME_WIDTH // 2
    results = {
        'black': {'found': False, 'error': 0, 'center_x': None},
        'red': {'found': False, 'error': 0, 'center_x': None},
        'blue': {'found': False, 'error': 0, 'center_x': None},
        'green': {'found': False, 'error': 0, 'center_x': None},
        'yellow': {'found': False, 'error': 0, 'center_x': None}
    }
    kernel = np.ones((5, 5), np.uint8)

    # Black
    mask_black = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_black:
        valid_contours_black = [cnt for cnt in contours_black if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours_black:
            largest_contour_black = max(valid_contours_black, key=cv2.contourArea)
            M_black = cv2.moments(largest_contour_black)
            if M_black["m00"] != 0:
                cx_black = int(M_black["m10"] / M_black["m00"])
                results['black']['found'] = True
                results['black']['error'] = cx_black - center_x_frame
                results['black']['center_x'] = cx_black

    # Red
    mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.erode(mask_red, kernel, iterations=1)
    mask_red = cv2.dilate(mask_red, kernel, iterations=1)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_red:
        valid_contours_red = [cnt for cnt in contours_red if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours_red:
            largest_contour_red = max(valid_contours_red, key=cv2.contourArea)
            M_red = cv2.moments(largest_contour_red)
            if M_red["m00"] != 0:
                cx_red = int(M_red["m10"] / M_red["m00"])
                results['red']['found'] = True
                results['red']['error'] = cx_red - center_x_frame
                results['red']['center_x'] = cx_red

    # Blue
    mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=1)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_blue:
        valid_contours_blue = [cnt for cnt in contours_blue if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours_blue:
            largest_contour_blue = max(valid_contours_blue, key=cv2.contourArea)
            M_blue = cv2.moments(largest_contour_blue)
            if M_blue["m00"] != 0:
                cx_blue = int(M_blue["m10"] / M_blue["m00"])
                results['blue']['found'] = True
                results['blue']['error'] = cx_blue - center_x_frame
                results['blue']['center_x'] = cx_blue

    # Green
    mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    mask_green = cv2.erode(mask_green, kernel, iterations=1)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_green:
        valid_contours_green = [cnt for cnt in contours_green if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours_green:
            largest_contour_green = max(valid_contours_green, key=cv2.contourArea)
            M_green = cv2.moments(largest_contour_green)
            if M_green["m00"] != 0:
                cx_green = int(M_green["m10"] / M_green["m00"])
                results['green']['found'] = True
                results['green']['error'] = cx_green - center_x_frame
                results['green']['center_x'] = cx_green

    # Yellow
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask_yellow = cv2.erode(mask_yellow, kernel, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, kernel, iterations=1)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_yellow:
        valid_contours_yellow = [cnt for cnt in contours_yellow if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if valid_contours_yellow:
            largest_contour_yellow = max(valid_contours_yellow, key=cv2.contourArea)
            M_yellow = cv2.moments(largest_contour_yellow)
            if M_yellow["m00"] != 0:
                cx_yellow = int(M_yellow["m10"] / M_yellow["m00"])
                results['yellow']['found'] = True
                results['yellow']['error'] = cx_yellow - center_x_frame
                results['yellow']['center_x'] = cx_yellow

    return results

# --------------------------------------------------------------------------------
#                           ARROW / SHAPE DETECTION
# --------------------------------------------------------------------------------

def get_all_children(hierarchy, parent_idx):
    children = []
    current_idx = parent_idx
    while current_idx != -1:
        child_idx = hierarchy[0][current_idx][2]
        while child_idx != -1:
            children.append(child_idx)
            children += get_all_children(hierarchy, child_idx)
            child_idx = hierarchy[0][child_idx][0]
        current_idx = hierarchy[0][current_idx][0]
    return children

def is_arrow(contour):
    try:
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 4:
            return False
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return False
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return False
        has_tip = any(d[0, 3] > 15 for d in defects)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        return has_tip and (aspect_ratio > 2 or aspect_ratio < 0.1)
    except:
        return False

def find_arrow_direction(contour):
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None or defects.shape[0] == 0:
        return "unknown"
    tip_point = None
    max_distance = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > max_distance:
            max_distance = d
            tip_point = contour[f][0]
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        cy = y + h // 2
    direction = "unknown"
    if tip_point is not None:
        dx = tip_point[0] - cx
        dy = tip_point[1] - cy
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        if angle < 45 or angle >= 315:
            direction = "right"
        elif 45 <= angle < 135:
            direction = "up"
        elif 135 <= angle < 225:
            direction = "left"
        else:
            direction = "down"
    return direction

def detect_arrow_shape(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or hierarchy is None:
        return "No object"
    outer_contours = [(c, i) for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
    if not outer_contours:
        return "No object"
    main_contour, main_idx = max(outer_contours, key=lambda x: cv2.contourArea(x[0]))
    area = cv2.contourArea(main_contour)
    if area < 1500:
        return "No object"
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    vertices = len(approx)
    shape = "unknown"
    if vertices == 3:
        shape = "triangle"
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        if 0.95 <= ar <= 1.05:
            shape = "square"
        else:
            shape = "rectangle"
    elif vertices == 5:
        shape = "pentagon"
    elif vertices == 6:
        shape = "hexagon"
    else:
        perimeter = cv2.arcLength(main_contour, True)
        if perimeter == 0:
            return "No object"
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.75:
            shape = "circle"
        elif 0.6 <= circularity <= 0.75:
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                ratio = area / hull_area
                if 0.6 <= ratio <= 0.9:
                    shape = "3-quarter circle"
                else:
                    shape = "circle"
    children_indices = get_all_children(hierarchy, main_idx)
    arrow_found = False
    arrow_direction = None
    for child_idx in children_indices:
        child = contours[child_idx]
        if cv2.contourArea(child) > 100:
            if is_arrow(child):
                arrow_found = True
                arrow_direction = find_arrow_direction(child)
                break
    if arrow_found:
        return f"{shape} + arrow ({arrow_direction})"
    else:
        return shape

# --------------------------------------------------------------------------------
#                           ORB-BASED SYMBOL DETECTION
# --------------------------------------------------------------------------------

def load_orb_symbols():
    symbol_paths = {
        "arrow right": "/home/pi/Downloads/data/arror.jpg",
        "arrow left":  "/home/pi/Downloads/data/leftArror.jpg",
        "arrow down":  "/home/pi/Downloads/data/downArror.jpg",
        "arrow up":    "/home/pi/Downloads/data/upArror.jpg",
        "circle arrow up":    "/home/pi/Downloads/data/cup.jpg",
        "circle arrow down":  "/home/pi/Downloads/data/cdown.jpg",
        "circle arrow left":  "/home/pi/Downloads/data/cleft.jpg",
        "circle arrow right": "/home/pi/Downloads/data/cright.jpg",
        "partial circle":     "/home/pi/Downloads/data/partialCircle.jpg",
        "distance":           "/home/pi/Downloads/data/dis.jpg",
        "face":               "/home/pi/Downloads/data/face.jpg",
        "hand stop":          "/home/pi/Downloads/data/handStop.jpg",
        "stop":               "/home/pi/Downloads/data/stop.jpg",
        "partial circle pc1": "/home/pi/Downloads/data2/pc1.JPG",
        "partial circle pc2": "/home/pi/Downloads/data2/pc2.JPG",
        "partial circle pc3": "/home/pi/Downloads/data2/pc3.JPG",
        "partial circle pc4": "/home/pi/Downloads/data2/pc4.JPG",
        "partial circle pc5": "/home/pi/Downloads/data2/pc5.JPG",
        "partial circle pc6": "/home/pi/Downloads/data2/pc6.JPG",
        "partial circle pc7": "/home/pi/Downloads/data2/pc7.JPG",
        "circle arrow up #2":   "/home/pi/Downloads/data2/cup1.JPG",
        "circle arrow up #3":   "/home/pi/Downloads/data2/cup2.JPG",
        "circle arrow down #2": "/home/pi/Downloads/data2/cdown1.JPG",
        "circle arrow down #3": "/home/pi/Downloads/data2/cdown2.JPG",
        "circle arrow left #2": "/home/pi/Downloads/data2/cleft1.JPG",
        "circle arrow left #3": "/home/pi/Downloads/data2/cleft2.JPG",
        "circle arrow right #2": "/home/pi/Downloads/data2/cright1.JPG",
        "circle arrow right #3": "/home/pi/Downloads/data2/cright2.JPG",
    }
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    symbol_features = {}
    for name, path in symbol_paths.items():
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load symbol image: {path}")
            continue
        kp_symbol, des_symbol = orb.detectAndCompute(image, None)
        if des_symbol is not None:
            symbol_features[name] = (kp_symbol, des_symbol)
    return orb, bf, symbol_features

def orb_detect_symbols(frame_bgr, orb, bf, symbol_features, prev_detected_symbol=None):
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    if des_frame is None:
        if prev_detected_symbol:
            print("No symbol detected")
        return None, None
    matches_dict = {}
    for name, (kp_sym, des_sym) in symbol_features.items():
        matches = bf.knnMatch(des_sym, des_frame, k=2)
        good_matches = [m for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        matches_dict[name] = len(good_matches)
    if not matches_dict:
        return None, None
    max_matches = max(matches_dict.values())
    detection_threshold = 15
    if max_matches > detection_threshold:
        candidate_symbol = max(matches_dict, key=matches_dict.get)
        if candidate_symbol != prev_detected_symbol:
            print(f"Detected symbol: {candidate_symbol}")
            return candidate_symbol, candidate_symbol
        else:
            return candidate_symbol, candidate_symbol
    else:
        if prev_detected_symbol:
            print("No symbol detected")
        return None, None

# --------------------------------------------------------------------------------
#                                MAIN PROGRAM
# --------------------------------------------------------------------------------

def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    set_servo_angle_simple(servo_pwm, 90)
    orb, bf, symbol_features = load_orb_symbols()
    prev_symbol = None

    # Set priority colors
    priority_input = input("Enter priority colors (e.g., 'br' for blue and red, 'r' for red only): ")
    priority_colors = []
    if 'b' in priority_input:
        priority_colors.append('blue')
    if 'r' in priority_input:
        priority_colors.append('red')
    if 'g' in priority_input:
        priority_colors.append('green')
    if 'y' in priority_input:
        priority_colors.append('yellow')
    priority_colors = list(set(priority_colors))[:2]  # Up to 2 unique colors
    print(f"Priority colors set to: {priority_colors}")

    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None

    print("System started. Press 'q' or Ctrl+C to exit.")

    try:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            line_detections = detect_lines(frame_bgr)

            # Determine target based on priority
            detected_priority = [color for color in priority_colors if line_detections[color]['found']]
            if detected_priority:
                target_color = min(detected_priority, key=lambda c: abs(line_detections[c]['error']))
                target_error = line_detections[target_color]['error']
                target_found = True
            elif line_detections['black']['found']:
                target_color = 'black'
                target_error = line_detections['black']['error']
                target_found = True
            else:
                target_color = None
                target_error = 0
                target_found = False

            # State Machine
            if state == "NORMAL":
                if target_found:
                    print(f"Following {target_color} line. Error: {target_error}")
                    if target_error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                    elif target_error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                    else:
                        move_forward(right_pwm, left_pwm)
                else:
                    print("Line lost. Reversing...")
                    stop_motors(right_pwm, left_pwm)
                    time.sleep(0.1)
                    state = "REVERSING"
                    reverse_start_time = time.time()
                    move_backward(right_pwm, left_pwm, REVERSE_SPEED)

            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    print("Reversing complete. Beginning scan...")
                    state = "SCANNING"
                    current_scan_index = 0
                    set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()

            elif state == "SCANNING":
                frame = picam2.capture_array()
                frame_bgr_scan = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                scan_detections = detect_lines(frame_bgr_scan)
                detected_priority_in_scan = [color for color in priority_colors if scan_detections[color]['found']]
                if detected_priority_in_scan:
                    scan_target_color = detected_priority_in_scan[0]
                    line_found_in_scan = True
                elif scan_detections['black']['found']:
                    scan_target_color = 'black'
                    line_found_in_scan = True
                else:
                    scan_target_color = None
                    line_found_in_scan = False

                if line_found_in_scan:
                    detected_scan_angle = SCAN_ANGLES[current_scan_index]
                    print(f"{scan_target_color} line found during scan at angle {detected_scan_angle}")
                    state = "TURNING"
                elif time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    current_scan_index += 1
                    if current_scan_index < len(SCAN_ANGLES):
                        print(f"Scanning at angle: {SCAN_ANGLES[current_scan_index]}")
                        set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                        scan_start_time = time.time()
                    else:
                        print("No line found in full scan. Reversing again...")
                        state = "REVERSING"
                        move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                        reverse_start_time = time.time()

            elif state == "TURNING":
                if detected_scan_angle is not None:
                    print(f"Turning based on scanned angle: {detected_scan_angle}")
                    turn_with_scanned_angle(detected_scan_angle, servo_pwm, right_pwm, left_pwm)
                    detected_scan_angle = None
                print("Turning complete. Resetting servo and returning to NORMAL.")
                set_servo_angle_simple(servo_pwm, 90)
                state = "NORMAL"

            # Arrow/Shape Detection
            shape_info = detect_arrow_shape(frame_bgr)
            if shape_info and shape_info != "No object":
                cv2.putText(frame_bgr, shape_info, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ORB Symbol Detection
            best_symbol, prev_symbol = orb_detect_symbols(frame_bgr, orb, bf, symbol_features, prev_symbol)
            if best_symbol:
                cv2.putText(frame_bgr, best_symbol, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Display status
            status_text = f"State: {state} | Priority: {priority_colors} | Following: {target_color if target_found else 'None'}"
            cv2.putText(frame_bgr, status_text, (10, FRAME_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Combined System", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram stopped by user (KeyboardInterrupt).")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle_simple(servo_pwm, 90)
        cv2.destroyAllWindows()
        picam2.stop()
        GPIO.cleanup()
        print("Resources released. Exiting...")

# --------------------------------------------------------------------------------
#                                ENTRY POINT
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    main()