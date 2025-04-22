import RPi.GPIO as GPIO
import time
import numpy as np
import cv2
from picamera2 import Picamera2

#GPIO pins
IN1, IN2 = 23, 24
IN3, IN4 = 27, 22
ENA, ENB = 13, 12
encoderPinRight, encoderPinLeft = 18, 19
ServoMotor = 26

#Constants
WHEEL_DIAMETER = 4.05
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER

#Servo motor parameters
SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
SERVO_FREQ = 50

#Line following parameters
BASE_SPEED = 50 #Base motor speed
TURN_SPEED = 70 #Speed for pivot turning
SLOW_SPEED = 30 #Speed for decel
MIN_CONTOUR_AREA = 300 #Minimum area for value contours
FRAME_WIDTH = 320 #Camera frame width
FRAME_HEIGHT = 240 #Camera frame height
ROI_HEIGHT = 50 #Region of interest for decel consideration
ROI_OFFSET = 270 #Offset for ROI, starting from the bottom

#Threshold for turning (error value)
TURN_THRESHOLD = 60

#Recovery parameters
REVERSE_DURATION = 0.3 
REVERSE_SPEED = 40

#Scanning angles, 90 is middle
SCAN_ANGLES = [90, 45, 135]
SCAN_TIME_PER_ANGLE = 0.5

#Store encoder counts
right_counter = 0
left_counter = 0

#Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

#GPIO setup
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

# Initialize camera
def setup_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam2.configure(config)
    picam2.start()
    return picam2

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

def move_forward(right_pwm, left_pwm, variation):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    if variation:
        right_pwm.ChangeDutyCycle(SLOW_SPEED)
        left_pwm.ChangeDutyCycle(SLOW_SPEED)
    else:
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

#Modified line detection function:
#Returns error, line_found, and intersection flag
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

def upgred_detect_line(frame): #returns the same as detect line, and an extra variation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 120])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    roi = frame[FRAME_HEIGHT-ROI_OFFSET: FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0: FRAME_WIDTH]
    roi_hsv = hsv[FRAME_HEIGHT-ROI_OFFSET: FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0: FRAME_WIDTH]
    mask_black_roi = cv2.inRange(roi_hsv, lower_black, upper_black)
    mask_black_roi = cv2.erode(mask_black_roi, kernel, iterations=1)
    mask_black_roi = cv2.dilate(mask_black_roi, kernel, iterations=1)

    lineContours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Used to follow the line

    boxContours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Used to detect slight variations in the line to decelerate

    cv2.rectangle(frame, (0, FRAME_HEIGHT-ROI_OFFSET), (FRAME_WIDTH, FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT), (0, 255, 0), 2) #Creates the rectangle for the ROI

    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, 0), (center_x, FRAME_HEIGHT), (0, 0, 255), 2)

    intersection = False

    variation = True

    if boxContours:
        largest_contour = max(boxContours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            M = cv2.moments(largest_contour)

            if M["M00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)

                variation = cx - center_x

                if abs(variation) < 40:
                    variation = False
                
                cv2.line(roi, (center_x, cy), (cx, cy), (255, 0, 0), 2)   

    if lineContours: #if black line detected, for following line
        valid_contours = [cnt for cnt in lineContours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

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
    return 0, False, intersection, variation


#Main function
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
        while True:
            frame = picam2.capture_array()
            # Adjusted to unpack three return values.
            error, line_found, intersection, variation = upgred_detect_line(frame)
            cv2.imshow("Line Follower", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if state == "NORMAL":
                if line_found:
                    if error > TURN_THRESHOLD:
                        pivot_turn_right(right_pwm, left_pwm)
                        print("Pivot Turning Right")
                    elif error < -TURN_THRESHOLD:
                        pivot_turn_left(right_pwm, left_pwm)
                        print("Pivot Turning Left")
                    else:
                        if variation:
                            move_forward(right_pwm, left_pwm, True)
                        else:
                            move_forward(right_pwm, left_pwm, False)
                        print("Moving Forward")
                else:
                    print("Line lost. Reversing...")
                    state = "REVERSING"
                    reverse_start_time = time.time()
                    move_backward(right_pwm, left_pwm, REVERSE_SPEED)
            
            elif state == "REVERSING":
                if time.time() - reverse_start_time >= REVERSE_DURATION:
                    stop_motors(right_pwm, left_pwm)
                    print("Beginning scan for line...")
                    state = "SCANNING"
                    current_scan_index = 0
                    # Set servo to first scan angle
                    set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                    scan_start_time = time.time()
            
            elif state == "SCANNING":
                if time.time() - scan_start_time >= SCAN_TIME_PER_ANGLE:
                    frame = picam2.capture_array()
                    error, line_found, intersection, variation = upgred_detect_line(frame)
                    # Check if an intersection is detected
                    if intersection:
                        print("Intersection detected. Centering servo to 90ï¿½ and adjusting.")
                        set_servo_angle_simple(servo_pwm, 90)
                        state = "NORMAL"
                    elif line_found:
                        detected_scan_angle = SCAN_ANGLES[current_scan_index]
                        print(f"Line detected during scan at servo angle: {detected_scan_angle}")
                        state = "TURNING"
                    else:
                        current_scan_index += 1
                        if current_scan_index < len(SCAN_ANGLES):
                            set_servo_angle_simple(servo_pwm, SCAN_ANGLES[current_scan_index])
                            scan_start_time = time.time()
                        else:
                            print("No line found during scan. Reversing again...")
                            state = "REVERSING"
                            move_backward(right_pwm, left_pwm, REVERSE_SPEED)
                            reverse_start_time = time.time()
            
            elif state == "TURNING":
                if detected_scan_angle is not None:
                    turn_with_scanned_angle(detected_scan_angle, servo_pwm, right_pwm, left_pwm)
                state = "NORMAL"
            
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
