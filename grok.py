#Code optimized by Grok, but not

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2

# --- GPIO Setup ---
GPIO.setmode(GPIO.BCM)

# Define motor control pins
IN1, IN2, IN3, IN4, ENA, ENB = 23, 24, 27, 22, 12, 13
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# Initialize PWM for motor speed control
pwm_ena = GPIO.PWM(ENA, 900)  # PWM frequency: 900 Hz
pwm_enb = GPIO.PWM(ENB, 900)
pwm_ena.start(0)
pwm_enb.start(0)

# Set initial motor direction to forward
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.HIGH)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.HIGH)

# --- Camera Setup ---
camera = Picamera2()
camera.resolution = (320, 240)  # Image size: 320x240 pixels
camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
camera.framerate = 60
camera.start()

# --- PID Controller Class ---
class PID:
    def __init__(self, kp, ki, kd):
        """Initialize PID controller with proportional, integral, and derivative gains."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.previous_time = time.time()
        
    def compute(self, error):
        """Compute PID output based on current error."""
        current_time = time.time()
        time_delta = current_time - self.previous_time
        if time_delta == 0:
            time_delta = 1e-6  # Prevent division by zero
        self.integral += error * time_delta
        derivative = (error - self.previous_error) / time_delta
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error
        self.previous_time = current_time
        return output

# Initialize PID controller and base parameters
pid = PID(kp=0.6, ki=0.05, kd=0.03)  # Tune these gains as needed
base_speed = 0.4  # Base motor speed (0 to 1)
lastLine = 160  # Initial line position (center of 320px wide image)

# --- Main Loop ---
try:
    while True:
        # Capture frame from camera
        frame = camera.capture_array()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to detect white line on black background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological opening to remove noise
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Define region of interest (ROI) at bottom of image
        roi = binary[200:240, :]  # Rows 200-240, all columns
        
        # Find contours in the ROI
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select the largest contour (assumed to be the line)
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum area to filter noise
                # Calculate centroid of the contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:  # Ensure non-zero area
                    cx = M["m10"] / M["m00"]  # Centroid x-coordinate
                    averageLinePosition = cx
                    lastLine = cx  # Update last known position
                else:
                    averageLinePosition = lastLine
            else:
                averageLinePosition = lastLine  # Use last position if contour too small
        else:
            # No contours detected: reverse the car
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
            pwm_ena.ChangeDutyCycle(30)  # 30% duty cycle
            pwm_enb.ChangeDutyCycle(30)
            continue  # Skip to next iteration
        
        # Calculate error and PID correction
        positionDelta = (160 - averageLinePosition) / 160  # Normalized error from center
        correction = pid.compute(positionDelta)
        print(f"Correction: {correction}")  # Debugging output
        
        # Calculate motor speeds
        right_speed = base_speed + correction
        left_speed = base_speed - correction
        
        # Control left motor
        if left_speed > 0:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)  # Forward
            left_duty = min(left_speed * 75, 100)  # Scale to 0-100%
        elif left_speed < 0:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)  # Reverse
            left_duty = min(-left_speed * 75, 100)
        else:
            left_duty = 0
        pwm_enb.ChangeDutyCycle(left_duty)
        
        # Control right motor
        if right_speed > 0:
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)  # Forward
            right_duty = min(right_speed * 75, 100)
        elif right_speed < 0:
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)  # Reverse
            right_duty = min(-right_speed * 75, 100)
        else:
            right_duty = 0
        pwm_ena.ChangeDutyCycle(right_duty)

except KeyboardInterrupt:
    print("Stopped by user")
    # Cleanup
    pwm_ena.ChangeDutyCycle(0)
    pwm_enb.ChangeDutyCycle(0)
    GPIO.cleanup()