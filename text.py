import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

# Define GPIO pins (from Main_Motor.py)
IN1, IN2 = 22, 27         # Left motor control
IN3, IN4 = 17, 4          # Right motor control
ENA, ENB = 13, 12         # PWM pins for motors
encoderPinRight = 23      # Right encoder
encoderPinLeft = 24       # Left encoder
ServoMotor = 18           # Servo motor PWM for the camera

# Constants (to be calibrated)
WHEEL_DIAMETER = 4.05  # cm
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER  # cm

# Variables to store encoder counts
right_counter = 0
left_counter = 0

# Line following parameters
BASE_SPEED = 50           # Base motor speed (0-100)
MIN_CONTOUR_AREA = 1000   # Minimum area for valid contours
FRAME_WIDTH = 640         # Camera frame width
FRAME_HEIGHT = 480        # Camera frame height
ROI_HEIGHT = 150          # Height of region of interest
ROI_OFFSET = 250          # Offset from bottom of frame

# PID Controller parameters
kp = 0.5                  # Proportional gain
ki = 0.01                 # Integral gain
kd = 0.2                  # Derivative gain
previous_error = 0
integral = 0

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
    
    # Set up PWM
    right_pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
    left_pwm = GPIO.PWM(ENB, 1000)
    
    right_pwm.start(0)
    left_pwm.start(0)
    
    return right_pwm, left_pwm

# Motor control functions (modified from Main_Motor.py)
def set_motor_speeds(right_pwm, left_pwm, right_speed, left_speed):
    # Ensure speeds are within 0-100 range
    right_speed = max(0, min(100, right_speed))
    left_speed = max(0, min(100, left_speed))
    
    # Set motor directions based on sign
    if right_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        right_speed = abs(right_speed)
    
    if left_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        left_speed = abs(left_speed)
    
    # Apply PWM speeds
    right_pwm.ChangeDutyCycle(right_speed)
    left_pwm.ChangeDutyCycle(left_speed)

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
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.start()
    return picam2

# Line detection function
def detect_line(frame):
    global previous_error, integral
    
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define specific range for black color detection
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    
    # Apply ROI (Region of Interest) - only look at the lower portion of the frame
    roi = frame[FRAME_HEIGHT-ROI_OFFSET:FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0:FRAME_WIDTH]
    roi_hsv = hsv[FRAME_HEIGHT-ROI_OFFSET:FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0:FRAME_WIDTH]
    
    # Create mask for black regions
    mask_black = cv2.inRange(roi_hsv, lower_black, upper_black)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw ROI rectangle on original frame
    cv2.rectangle(frame, 
                 (0, FRAME_HEIGHT-ROI_OFFSET), 
                 (FRAME_WIDTH, FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT), 
                 (0, 255, 0), 2)
    
    # Draw center reference line
    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, FRAME_HEIGHT-ROI_OFFSET), 
             (center_x, FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT), (0, 0, 255), 2)
    
    # Process contours
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            # Calculate the moments of the contour
            M = cv2.moments(largest_contour)
            
            # Draw the contour
            cv2.drawContours(roi, [largest_contour], -1, (0, 255, 0), 2)
            
            # If moment is valid, calculate centroid
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw the centroid
                cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)
                
                # Calculate the error (distance from center)
                error = cx - center_x
                
                # Draw line from center to centroid
                cv2.line(roi, (center_x, cy), (cx, cy), (255, 0, 0), 2)
                
                # Display error value
                cv2.putText(frame, f"Error: {error}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Apply PID control
                return apply_pid(error)
    
    # Return zero correction if no valid line detected
    return 0

# PID controller function
def apply_pid(error):
    global previous_error, integral
    
    # Calculate PID components
    proportional = kp * error
    integral += ki * error
    derivative = kd * (error - previous_error)
    
    # Limit integral windup
    integral = max(-50, min(50, integral))
    
    # Calculate the correction
    correction = proportional + integral + derivative
    
    # Update previous error
    previous_error = error
    
    return correction

# Main function
def main():
    # Initialize GPIO and PWM
    right_pwm, left_pwm = setup_gpio()
    
    # Initialize camera
    picam2 = setup_camera()
    
    print("Line follower started. Press Ctrl+C to stop.")
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Detect line and get steering correction
            correction = detect_line(frame)
            
            # Calculate motor speeds based on correction
            right_speed = BASE_SPEED - correction
            left_speed = BASE_SPEED + correction
            
            # Apply motor speeds
            set_motor_speeds(right_pwm, left_pwm, right_speed, left_speed)
            
            # Display information
            cv2.putText(frame, f"Left: {left_speed:.1f} | Right: {right_speed:.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Show the frame
            cv2.imshow("Line Follower", frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nProgram stopped by user")
    finally:
        # Cleanup
        stop_motors(right_pwm, left_pwm)
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Resources released")

if _name_ == "_main_":
    main()