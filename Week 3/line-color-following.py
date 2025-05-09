import RPi.GPIO as gpio
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
SCAN_TIME_PER_ANGLE = 0.5

right_counter = 0
left_counter = 0

# Encoder callback functions
def right_encoder_callback(channel):
    global right_counter
    right_counter += 1

def left_encoder_callback(channel):
    global left_counter
    left_counter += 1

def setServoAngleSimple(servoPWM, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle / 180.0) * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)
    servoPWM.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servoPWM.ChangeDutyCycle(0)  # Stop sending PWM signal to the servo

def pivotturnRight(rightPWM, leftPWM):
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.HIGH)
    rightPWM.ChangeDutyCycle(TURN_SPEED)
    leftPWM.ChangeDutyCycle(TURN_SPEED)

def pivotturnLeft(rightPWM, leftPWM):
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)
    rightPWM.ChangeDutyCycle(TURN_SPEED)
    leftPWM.ChangeDutyCycle(TURN_SPEED)

def moveForward(rightPWM, leftPWM):
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.LOw)
    gpio.output(IN4, gpio.HIGH)
    rightPWM.ChangeDutyCycle(BASE_SPEED)
    leftPWM.ChangeDutyCycle(BASE_SPEED)

def moveBackward(rightPWM, leftPWM, speed):
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)
    rightPWM.ChangeDutyCycle(speed)
    leftPWM.ChangeDutyCycle(speed)

def stop(rightPWM, leftPWM):
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.LOW)
    rightPWM.ChangeDutyCycle(0)
    leftPWM.ChangeDutyCycle(0)

def setupCam():
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    cam.start()
    return cam

# GPIO setup
def setupGPIO():
    gpio.setmode(gpio.BCM)
    gpio.setup([IN1, IN2, IN3, IN4, ENA, ENB], gpio.OUT)
    gpio.setup([encoderPinRight, encoderPinLeft], gpio.IN, pull_up_down=gpio.PUD_UP)
    gpio.add_event_detect(encoderPinRight, gpio.RISING, callback=right_encoder_callback)
    gpio.add_event_detect(encoderPinLeft, gpio.RISING, callback=left_encoder_callback)
    gpio.setup(ServoMotor, gpio.OUT)

    rightPWM = gpio.PWM(ENA, 1000)  # 1kHz frequency
    leftPWM = gpio.PWM(ENB, 1000)   # 1kHz frequency
    rightPWM.start(0)               # Start with 0% duty cycle
    leftPWM.start(0)                # Start with 0% duty cycle

    servoPWM = gpio.PWM(ServoMotor, 50)  # 50Hz frequency for servo
    servoPWM.start(0)                # Start with 0% duty cycle

    return rightPWM, leftPWM, servoPWM

def turnWithScannedAngle(scannedAngle, servoPWM, rightPWM, leftPWM):
    turnTime = abs(scannedAngle - 90) / 45.0
    if scannedAngle > 90:
        gpio.output(IN1, gpio.LOW)
        gpio.output(IN2, gpio.HIGH)
        gpio.output(IN3, gpio.HIGH)
        gpio.output(IN4, gpio.LOW)
        rightPWM.ChangeDutyCycle(TURN_SPEED)
        leftPWM.ChangeDutyCycle(TURN_SPEED)
    elif scannedAngle < 90:
        gpio.output(IN1, gpio.HIGH)
        gpio.output(IN2, gpio.LOW)
        gpio.output(IN3, gpio.LOW)
        gpio.output(IN4, gpio.HIGH)
        rightPWM.ChangeDutyCycle(TURN_SPEED)
        leftPWM.ChangeDutyCycle(TURN_SPEED)
    else:
        print("Detected angle 90: No pivot required")
        return

    time.sleep(turnTime)
    stop(rightPWM, leftPWM)

"""
def calibrateColor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[FRAME_HEIGHT // 2, FRAME_WIDTH // 2]
    print(f"HSV value at center: H={h}, S={s}, V={v}")

    # Define color ranges based on the center pixel's HSV value

    global lower_red, upper_red
    global lower_blue, upper_blue
    global lower_green, upper_green
    global lower_yellow, upper_yellow
    global lower_black, upper_black

    while True:
        cv2.imshow('calibrating', frame)
        color = input("Press 'r' to calibrate red, 'b' for blue, 'g' for green, 'y' for yellow, 'k' for black, or 'q' to quit.")
        if color == 'r':
            lower_red = np.array([h - 10, s - 30, v - 30])
            upper_red = np.array([h + 10, s + 30, v + 30])
            print(f"Red range: {lower_red}, {upper_red}")
        elif color == 'b':
            lower_blue = np.array([h - 10, s - 30, v - 30])
            upper_blue = np.array([h + 10, s + 30, v + 30])
            print(f"Blue range: {lower_blue}, {upper_blue}")
        elif color == 'g':
            lower_green = np.array([h - 10, s - 30, v - 30])
            upper_green = np.array([h + 10, s + 30, v + 30])
            print(f"Green range: {lower_green}, {upper_green}")
        elif color == 'y':
            lower_yellow = np.array([h - 10, s - 30, v - 30])
            upper_yellow = np.array([h + 10, s + 30, v + 30])
            print(f"Yellow range: {lower_yellow}, {upper_yellow}")
        elif color == 'k':
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, s + 50, v + 50])
            print(f"Black range: {lower_black}, {upper_black}")
        elif color == 'q':
            break

def calibrateColor(event, x, y, param):
    global lower_red, upper_red
    global lower_green, upper_green
    global lower_blue, upper_blue
    global lower_yellow, upper_yellow
    global lower_black, upper_black

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_frame = param["hsv_frame"]
        h, s, v = hsv_frame[y, x]
        color_name = param["color_name"]
        print(f"{color_name} HSV value at ({x},{y}): H={h}, S={s}, V={v}")

        param["h_values"].append(h)
        param["s_values"].append(s)
        param["v_values"].append(v)

        if len(param["h_values"]) > 0:
            h_min, h_max = max(0, min(param["h_values"]) - 10), min(180, max(param["h_values"]) + 10)
            s_min, s_max = max(0, min(param["s_values"]) - 30), min(255, max(param["s_values"]) + 30)
            v_min, v_max = max(0, min(param["v_values"]) - 30), min(255, max(param["v_values"]) + 30)

            print(f"Suggested range for {color_name}:")
            print(f"lower_{color_name} = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_{color_name} = np.array([{h_max}, {s_max}, {v_max}])")

            if color_name == "red" and h_min < 10 and h_max > 170:
                print("Red color wraps around, consider using two ranges:")
                print(f"lower_red1 = np.array([0, {s_min}, {v_min}])")
                print(f"upper_red1 = np.array([10, {s_max}, {v_max}])")
                print(f"lower_red2 = np.array([170, {s_min}, {v_min}])")
                print(f"upper_red2 = np.array([180, {s_max}, {v_max}])")
"""

def calibrateColor(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsvFrame = param['frame']
        h, s, v = hsvFrame[y, x]
        print(f"HSV value at ({x},{y}): H={h}, S={s}, V={v}")

        param['lowerBound'] = np.array([h - 10, s - 30, v - 30])
        param['upperBound'] = np.array([h + 10, s + 30, v + 30])
        print(f"Suggested range: {param['lowerBound']}, {param['upperBound']}")


def detectLine(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    centerX = FRAME_WIDTH // 2
    intersection = False

    kernel = np.ones((5, 5), np.uint8)

    global allContours
    global upper_red, lower_red
    global upper_blue, lower_blue
    global upper_green, lower_green
    global upper_yellow, lower_yellow
    global upper_black, lower_black

    allContours = []
    if len(priorityColors) > 0:  # User wants to follow colored line
        if 'red' in priorityColors:
            maskRed = cv2.inRange(hsv, lower_red, upper_red)
            maskRed = cv2.erode(maskRed, kernel, iterations=1)
            maskRed = cv2.dilate(maskRed, kernel, iterations=1)
            contoursRed, _ = cv2.findContours(maskRed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contoursRed:
                validContoursRed = [cnt for cnt in contoursRed if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
                if len(validContoursRed) > 1:
                    intersection = True
                if validContoursRed:
                    largestContour = max(validContoursRed, key=cv2.contourArea)
                    area = cv2.contourArea(largestContour)
                    M = cv2.moments(largestContour)
                    cv2.drawContours(frame, [largestContour], -1, (0, 255, 0), 2)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        error = cx - centerX
                        cv2.line(frame, (centerX, cy), (cx, cy), (255, 0, 0), 2)
                        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        allContours.append = [error, True, intersection]
            
        if 'blue' in priorityColors:
            maskBlue = cv2.inRange(hsv, lower_blue, upper_blue)
            maskBlue = cv2.erode(maskBlue, kernel, iterations=1)
            maskBlue = cv2.dilate(maskBlue, kernel, iterations=1)
            contoursBlue, _ = cv2.findContours(maskBlue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contoursBlue:
                validContoursBlue = [cnt for cnt in contoursBlue if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
                if len(validContoursBlue) > 1:
                    intersection = True
                if validContoursBlue:
                    largestContour = max(validContoursBlue, key=cv2.contourArea)
                    area = cv2.contourArea(largestContour)
                    M = cv2.moments(largestContour)
                    cv2.drawContours(frame, [largestContour], -1, (0, 255, 0), 2)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        error = cx - centerX
                        cv2.line(frame, (centerX, cy), (cx, cy), (255, 0, 0), 2)
                        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        allContours.append = [error, True, intersection]
        
        if 'green' in priorityColors:
            maskGreen = cv2.inRange(hsv, lower_green, upper_green)
            maskGreen = cv2.erode(maskGreen, kernel, iterations=1)
            maskGreen = cv2.dilate(maskGreen, kernel, iterations=1)
            contoursGreen, _ = cv2.findContours(maskGreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contoursGreen:
                validContoursGreen = [cnt for cnt in contoursGreen if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
                if len(validContoursGreen) > 1:
                    intersection = True
                if validContoursGreen:
                    largestContour = max(validContoursGreen, key=cv2.contourArea)
                    area = cv2.contourArea(largestContour)
                    M = cv2.moments(largestContour)
                    cv2.drawContours(frame, [largestContour], -1, (0, 255, 0), 2)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        error = cx - centerX
                        cv2.line(frame, (centerX, cy), (cx, cy), (255, 0, 0), 2)
                        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        allContours.append = [error, True, intersection]
        
        if 'yellow' in priorityColors:
            maskYellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            maskYellow = cv2.erode(maskYellow, kernel, iterations=1)
            maskYellow = cv2.dilate(maskYellow, kernel, iterations=1)
            contoursYellow, _ = cv2.findContours(maskYellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contoursYellow:
                validContoursYellow = [cnt for cnt in contoursYellow if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
                if len(validContoursYellow) > 1:
                    intersection = True
                if validContoursYellow:
                    largestContour = max(validContoursYellow, key=cv2.contourArea)
                    area = cv2.contourArea(largestContour)
                    M = cv2.moments(largestContour)
                    cv2.drawContours(frame, [largestContour], -1, (0, 255, 0), 2)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                        error = cx - centerX
                        cv2.line(frame, (centerX, cy), (cx, cy), (255, 0, 0), 2)
                        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        allContours.append = [error, True, intersection]
        
    maskBlack = cv2.inRange(hsv, lower_black, upper_black)
    maskBlack = cv2.erode(maskBlack, kernel, iterations=1)
    maskBlack = cv2.dilate(maskBlack, kernel, iterations=1)
    contoursBlack, _ = cv2.findContours(maskBlack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contoursBlack:
        validContoursBlack = [cnt for cnt in contoursBlack if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if len(validContoursBlack) > 1:
            intersection = True
        if validContoursBlack:
            largestContour = max(validContoursBlack, key=cv2.contourArea)
            area = cv2.contourArea(largestContour)
            M = cv2.moments(largestContour)
            cv2.drawContours(frame, [largestContour], -1, (0, 255, 0), 2)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] /M["m00"])
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                error = cx - centerX
                cv2.line(frame, (centerX, cy), (cx, cy), (0, 255, 0), 2)
                cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                allContours.append = [error, True, intersection]

    if len(allContours) == 0:
        allContours[0] = [0, False, intersection]


cam = setupCam()
rightPWM, leftPWM, servoPWM = setupGPIO()
state = 'NORMAL'
reverseStartTime = 0
currentScanIndex = 0
scanStartTime = 0
detectedScanAngle = None

ifCalibrated = False

try:
    # Initialize the camera
    priorityInput = input("Enter the colors that you want to follow\n")
    priorityColors = []
    if 'b' in priorityInput:
        priorityColors.append("blue")
    if 'g' in priorityInput:    
        priorityColors.append("green")
    if 'r' in priorityInput:
        priorityColors.append("red")
    if 'y' in priorityInput:
        priorityColors.append("yellow")

    priorityColors = list(set(priorityColors))[:4]  # Limit to 4 unique colors
    print(f"Following colors: {priorityColors}")

    while True:
        frame = cam.capture_array()
        if not ifCalibrated:
            global upper_red, lower_red
            global upper_blue, lower_blue
            global upper_green, lower_green
            global upper_yellow, lower_yellow
            global upper_black, lower_black

            color = input('Press the color to calibrate: ')
            if color == 'r':
                cv2.setMouseCallback("Calibration", calibrateColor, param={'frame': frame, 'lowerBound': lower_red, 'upperBound': None})
            cv2.imshow("Calibration", frame)
            cv2.setMouseCallback("Calibration", calibrateColor, param={'frame': frame, 'lowerBound': None, 'upperBound': None})
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        detectLine(frame)
        global allContours
        cv2.imshow("Line follower", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if state == "NORMAL":
            if allContours[0][1]:
                if allContours[0][0] > TURN_THRESHOLD:
                    pivotturnRight(rightPWM, leftPWM)
                    print("Turning right")
                elif allContours[0][0] < -TURN_THRESHOLD:
                    pivotturnLeft(rightPWM, leftPWM)
                    print("Turning left")
                else:
                    moveForward(rightPWM, leftPWM)
                    print("Moving forward")
            else:
                print("Line lost. Reversing...")
                state = "REVERSING"
                reverseStartTime = time.time()
                moveBackward(rightPWM, leftPWM, REVERSE_SPEED)

        elif state == "REVERSING":
            if time.time() - reverseStartTime > REVERSE_DURATION:
                stop(rightPWM, leftPWM)
                state = "SCANNING"
                print("Reversing complete. Scanning.")
                currentScanIndex = 0
                setServoAngleSimple(servoPWM, SCAN_ANGLES[currentScanIndex])
                scanStartTime = time.time()

        elif state == "SCANNING":
            if time.time() - scanStartTime >= SCAN_TIME_PER_ANGLE:
                frame = cam.capture_array()
                detectLine(frame)
                if allContours[0][2]:
                    print("Intersection detected.")
                    setServoAngleSimple(servoPWM, 90)
                    state = "NORMAL"
                elif allContours[0][1]:
                    detectedScanAngle = SCAN_ANGLES[currentScanIndex]
                    print(f"Detected angle: {detectedScanAngle}")
                    state = "TURNING"
                else:
                    currentScanIndex += 1
                    if currentScanIndex < len(SCAN_ANGLES):
                        setServoAngleSimple(servoPWM, SCAN_ANGLES[currentScanIndex])
                        scanStartTime = time.time()
                    else:
                        print("No line detected. Reversing.")
                        state = "REVERSING"
                        moveBackward(rightPWM, leftPWM, REVERSE_SPEED)
                        reverseStartTime = time.time()
        
        elif state == "TURNING":
            if detectedScanAngle is not None:
                turnWithScannedAngle(detectedScanAngle, servoPWM, rightPWM, leftPWM)
            state = "NORMAL"
    
except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    stop(rightPWM, leftPWM)
    setServoAngleSimple(servoPWM, 90)
    gpio.cleanup()
    cam.close()
    cv2.destroyAllWindows()
    print("GPIO cleaned up and program terminated.")