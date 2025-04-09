import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins
IN1, IN2 = 23, 24         # Left motor control
IN3, IN4 = 27, 22          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors (ENA = Right, ENB = Left)
encoderPinRight = 18      # Right encoder
encoderPinLeft = 19      # Left encoder
ServoMotor = 26           # Servo motor PWM for the camera

# Constants
WHEEL_DIAMETER = 4.05      # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Servo motor parameters
SERVO_MIN_DUTY = 2.5       # Duty cycle for 0 degrees
SERVO_MAX_DUTY = 12.5      # Duty cycle for 180 degrees
SERVO_FREQ = 50            # 50Hz frequency for servo

# Line following parameters
BASE_SPEED = 50           # Base motor speed (0-100) original 40
TURN_SPEED = 65            # Speed for pivot turns (0-100) original 50
MIN_CONTOUR_AREA = 250    # Minimum area for valid contours
FRAME_WIDTH = 320          # Camera frame width
FRAME_HEIGHT = 240         # Camera frame height

# Threshold for turning
TURN_THRESHOLD = 60        # Error threshold for pivoting

# Recovery parameters
REVERSE_DURATION = 0.3     # Seconds to reverse
REVERSE_SPEED = 40         # Speed when reversing

# Updated scanning angles: center at 90, right at 45, left at 135.
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5   # Seconds to wait per scan angle

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

# GPIO Setup
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Motor pins setup
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    
    # Encoder pins setup
    GPIO.setup(encoderPinRight, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(encoderPinLeft, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Set up encoder interrupts
    GPIO.add_event_detect(encoderPinRight, GPIO.RISING, callback=right_encoder_callback)
    GPIO.add_event_detect(encoderPinLeft, GPIO.RISING, callback=left_encoder_callback)
    
    # Set up PWM for motors
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    right_pwm.start(0)
    left_pwm.start(0)
    
    # Set up PWM for servo
    GPIO.setup(ServoMotor, GPIO.OUT)
    servo_pwm = GPIO.PWM(ServoMotor, SERVO_FREQ)
    servo_pwm.start(0)
    
    return right_pwm, left_pwm, servo_pwm

# Function to set servo angle (simple version for scanning and reset)
def set_servo_angle_simple(servo_pwm, angle):
    # Constrain angle
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.3)  # Allow time for movement
    servo_pwm.ChangeDutyCycle(0)

# New function: use servo tuning logic to perform a turn based on a scanned angle.
def turn_with_scanned_angle(scanned_angle, servo_pwm, right_pwm, left_pwm):
    # Calculate turn time: assume 45ï¿½ turn takes 1 second
    turn_time = abs(scanned_angle - 90) / 45.0
    if scanned_angle > 90:
        print(f"Detected angle {scanned_angle}: Pivoting LEFT for {turn_time:.2f} seconds")
        # For left pivot: right wheel forward, left wheel backward
        GPIO.output(IN1, GPIO.LOW)   # Left forward
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)   # Right backward
        GPIO.output(IN4, GPIO.LOW)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    elif scanned_angle < 90:
        print(f"Detected angle {scanned_angle}: Pivoting RIGHT for {turn_time:.2f} seconds")
        # For right pivot: left wheel forward, right wheel backward
        GPIO.output(IN1, GPIO.HIGH)   # Left forward
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)   # Right backward
        GPIO.output(IN4, GPIO.HIGH)
        right_pwm.ChangeDutyCycle(TURN_SPEED)
        left_pwm.ChangeDutyCycle(TURN_SPEED)
    else:
        print("Detected angle 90: No pivot required.")
        return

    time.sleep(turn_time)
    stop_motors(right_pwm, left_pwm)
    # Reset the servo to center
    print("Resetting servo to 90 degrees")
    set_servo_angle_simple(servo_pwm, 90)

# Motor control functions
def pivot_turn_right(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.HIGH)   # Left forward
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)   # Right backward
    GPIO.output(IN4, GPIO.HIGH)
    right_pwm.ChangeDutyCycle(TURN_SPEED)
    left_pwm.ChangeDutyCycle(TURN_SPEED)

def pivot_turn_left(right_pwm, left_pwm):
    GPIO.output(IN1, GPIO.LOW)   # Left forward
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)   # Right backward
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

def stop_motors(right_pwm, left_pwm):
    right_pwm.ChangeDutyCycle(0)
    left_pwm.ChangeDutyCycle(0)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

# Modified line detection function:
# Now returns error, line_found, and intersection flag.
def detect_line(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])  # Include dark gray
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)
    
    intersection = False
    if contours:
        # Filter out contours that are too small
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        # If two or more valid contours are detected, consider it an intersection.
        if len(valid_contours) >= 2:
            intersection = True
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            M = cv2.moments(largest_contour)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                error = cx - center_x
                cv2.line(frame, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return error, True, intersection
    return 0, False, intersection

# Main function
def main():
    right_pwm, left_pwm, servo_pwm = setup_gpio()
    picam2 = setup_camera()
    
    # Center the servo initially
    set_servo_angle_simple(servo_pwm, 90)
    
    # State variables
    state = "NORMAL"
    reverse_start_time = 0
    current_scan_index = 0
    scan_start_time = 0
    detected_scan_angle = None

    print("Line follower started. Press 'q' in the display window or Ctrl+C to stop.")
    
    try:
        move_forward(100, 100)
            
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        stop_motors(right_pwm, left_pwm)
        set_servo_angle_simple(servo_pwm, 90)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if __name__ == "__main__":
    main()
