import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# === GPIO Motor Pins ===
LEFT_MOTOR_FORWARD = 24
LEFT_MOTOR_BACKWARD = 23
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 27

# === Initialize GPIO ===
GPIO.setmode(GPIO.BCM)
GPIO.setup([LEFT_MOTOR_FORWARD, LEFT_MOTOR_BACKWARD, RIGHT_MOTOR_FORWARD, RIGHT_MOTOR_BACKWARD], GPIO.OUT)

# === PWM Motor Control ===
LEFT_PWM_FWD = GPIO.PWM(LEFT_MOTOR_FORWARD, 100)
LEFT_PWM_BWD = GPIO.PWM(LEFT_MOTOR_BACKWARD, 100)
RIGHT_PWM_FWD = GPIO.PWM(RIGHT_MOTOR_FORWARD, 100)
RIGHT_PWM_BWD = GPIO.PWM(RIGHT_MOTOR_BACKWARD, 100)

LEFT_PWM_FWD.start(0)
LEFT_PWM_BWD.start(0)
RIGHT_PWM_FWD.start(0)
RIGHT_PWM_BWD.start(0)

# === Speed Control Parameters ===
MAX_SPEED = 60  
MIN_SPEED = 20  
TURN_SPEED = 50  
TURN_THRESHOLD = 40  # Error threshold for sharp turns

# === PID Control Parameters ===
Kp = 1.0  
Ki = 0.01  
Kd = 0.1  

# === PID Variables ===
previous_error = 0
integral = 0

# === Initialize PiCamera2 ===
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.controls.FrameRate = 30
picam2.configure("preview")
picam2.start()
time.sleep(2)

def preprocess_frame(frame):
    """Convert frame to HSV and apply color thresholding for black detection."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask, hsv

def find_line(thresh):
    """Detects the main line using contours and finds its center."""
    roi = thresh[120:280, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            return cx
    return None

def control_motors(cx, frame_width):
    """Adjust motor speed based on line position using PID control."""
    global previous_error, integral

    center_x = frame_width // 2

    if cx is None:
        # Stop motors if no line detected
        LEFT_PWM_FWD.ChangeDutyCycle(0)
        LEFT_PWM_BWD.ChangeDutyCycle(0)
        RIGHT_PWM_FWD.ChangeDutyCycle(0)
        RIGHT_PWM_BWD.ChangeDutyCycle(0)
        return

    # Calculate error
    error = cx - center_x

    # Sharp Turn Handling
    if abs(error) > TURN_THRESHOLD:
        if error > 0 :
            # Left Turn: Left motor backward, Right motor forward
            LEFT_PWM_FWD.ChangeDutyCycle(0)
            LEFT_PWM_BWD.ChangeDutyCycle(TURN_SPEED)
            RIGHT_PWM_FWD.ChangeDutyCycle(TURN_SPEED)
            RIGHT_PWM_BWD.ChangeDutyCycle(0)
        else:
            # Right Turn: Left motor forward, Right motor backward
            LEFT_PWM_FWD.ChangeDutyCycle(TURN_SPEED)
            LEFT_PWM_BWD.ChangeDutyCycle(0)
            RIGHT_PWM_FWD.ChangeDutyCycle(0)
            RIGHT_PWM_BWD.ChangeDutyCycle(TURN_SPEED)
        return


    # PID Correction for Minor Adjustments
    integral += error
    derivative = error - previous_error
    correction = Kp * error + Ki * integral + Kd * derivative

    # Adjust motor speeds
    left_speed = MAX_SPEED - correction
    right_speed = MAX_SPEED + correction

    # Limit speed range
    left_speed = max(min(left_speed, MAX_SPEED), MIN_SPEED)
    right_speed = max(min(right_speed, MAX_SPEED), MIN_SPEED)

    # Apply speeds (both motors forward)
    LEFT_PWM_FWD.ChangeDutyCycle(left_speed)
    LEFT_PWM_BWD.ChangeDutyCycle(0)
    RIGHT_PWM_FWD.ChangeDutyCycle(right_speed)
    RIGHT_PWM_BWD.ChangeDutyCycle(0)

    # Debug info
    print(f"Error: {error}, Correction: {correction}, Left Speed: {left_speed}, Right Speed: {right_speed}")

    previous_error = error

try:
    while True:
        # Capture frame from camera
        frame = picam2.capture_array()

        # Preprocess frame (convert to HSV and threshold)
        mask, hsv = preprocess_frame(frame)

        # Find the line
        cx = find_line(mask)

        # Control motors based on line position
        control_motors(cx, frame.shape[1])

        # Display images
        cv2.imshow("RGB Preview", frame)
        cv2.imshow("HSV Preview", hsv)
        cv2.imshow("Thresholded Image", mask)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cv2.destroyAllWindows()
    LEFT_PWM_FWD.stop()
    LEFT_PWM_BWD.stop()
    RIGHT_PWM_FWD.stop()
    RIGHT_PWM_BWD.stop()
    GPIO.cleanup()
    picam2.close()