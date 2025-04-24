#Using stan logic, has no face recognition, only symbol recognition from tflite

import RPi.GPIO as gpio
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from tflite_runtime.interpreter import Interpreter

IN1, IN2 = 23, 24
IN3, IN4 = 27, 22
ENA, ENB = 13, 12
encoderPinRight = 18
encoderPinLeft = 19
servoMotor = 26

WHEEL_DIAMETER = 4.05
PULSES_PER_REVOLUTION = 20
WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER

SERVO_MIN_DUTY = 2.5
SERVO_MAX_DUTY = 12.5
SERVO_FREQ = 50

BASE_SPEED = 40
TURN_SPEED = 40
MIN_CONTOUR_AREA = 250
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

TURN_THRESHOLD = 60

REVERSE_DURATION = 0.3
REVERSE_SPEED = 40

SCAN_ANGLES = [90, 60, 120]
SCAN_TIME_PER_ANGLE = 0.5

rightCounter = 0
leftCounter = 0

def rightEncoderCallback(channel):
    global rightCounter
    rightCounter += 1

def leftEncoderCallback(channel):
    global leftCounter
    leftCounter += 1

def setServoAngleSimple(servoPWM, angle):
    if angle < 0:
        angle = 0
    elif angle > 180:
        angle = 180
    duty = SERVO_MIN_DUTY + (angle * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servoPWM.ChangeDutyCycle(duty)
    time.sleep(0.3)
    servoPWM.ChangeDutyCycle(0)

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
        print("Detected angle 90: No pivot required!")
        return

    time.sleep(turnTime)
    stopMotors(rightPWM, leftPWM)

def pivotTurnRight(rightPWM, leftPWM):
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.HIGH)
    rightPWM.ChangeDutyCycle(TURN_SPEED)
    leftPWM.ChangeDutyCycle(TURN_SPEED)

def pivotTurnLeft(rightPWM, leftPWM):
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)
    rightPWM.ChangeDutyCycle(TURN_SPEED)
    leftPWM.ChangeDutyCycle(TURN_SPEED)

def moveBackward(rightPWM, leftPWM, speed):
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)
    rightPWM.ChangeDutyCycle(speed)
    leftPWM.ChangeDutyCycle(speed)

def moveForward(rightPWM, leftPWM):
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.HIGH)
    rightPWM.ChangeDutyCycle(BASE_SPEED)
    leftPWM.ChangeDutyCycle(BASE_SPEED)

def stopMotors(rightPWM, leftPWM):
    rightPWM.ChangeDutyCycle(0)
    leftPWM.ChangeDutyCycle(0)
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.LOW)

def setupCam():
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    cam.start()
    return cam

def setupGPIO():
    gpio.setmode(gpio.BCM)
    gpio.setup([IN1, IN2, IN3, IN4, ENA, ENB], gpio.OUT)
    gpio.setup([encoderPinRight, encoderPinLeft], gpio.IN, pull_up_down=gpio.PUD_UP)
    gpio.setup(servoMotor, gpio.OUT)
    gpio.add_event_detect(encoderPinRight, gpio.RISING, callback=rightEncoderCallback)
    gpio.add_event_detect(encoderPinLeft, gpio.RISING, callback=leftEncoderCallback)

    rightPWM = gpio.PWM(ENA, 1000)
    leftPWM = gpio.PWM(ENB, 1000)
    rightPWM.start(0)
    leftPWM.start(0)

    servoPWM = gpio.PWM(servoMotor, SERVO_FREQ)
    servoPWM.start(0)

    return rightPWM, leftPWM, servoPWM

def setupModel():
    model_path = "/home/pi/Downloads/Week_3/imageDetection/model_unquant.tflite"
    label_path = "/home/pi/Downloads/Week_3/imageDetection/labels.txt"

    with open(label_path, 'r') as f:
        symbol_names = [line.strip() for line in f.readlines()]

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details, symbol_names

def detectLine(frame):
    global lowerRed1, upperRed1, lowerRed2, upperRed2
    global lowerBlack, upperBlack
    global lowerBlue, upperBlue
    global lowerGreen, upperGreen
    global lowerYellow, upperYellow
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    centerX = FRAME_WIDTH // 2
    intersection = False

    kernel = np.ones((5, 5), np.uint8)

    global allContours
    allContours = []
    if len(priorityColors) > 0:  # User wants to follow colored line
        if 'red' in priorityColors:
            maskRed = cv2.inRange(hsv, lowerRed1, upperRed1)
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
                        allContours.append([error, True, intersection])
            
        if 'blue' in priorityColors:
            maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)
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
                        allContours.append([error, True, intersection])
        
        if 'green' in priorityColors:
            maskGreen = cv2.inRange(hsv, lowerGreen, upperGreen)
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
                        allContours.append([error, True, intersection])
        
        if 'yellow' in priorityColors:
            maskYellow = cv2.inRange(hsv, lowerYellow, upperYellow)
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
                        allContours.append([error, True, intersection])

        try:
            for contours in allContours:
                if contours[1]:
                    print("Detected color line")
                    return
                else:
                    pass
        except Exception as e:
            print("No contours")
    
    if len(allContours) == 0:
        print("No contours detected.")
    if len(priorityColors) == 0:
        print("No line chosen")
    maskBlack = cv2.inRange(hsv, lowerBlack, upperBlack)
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
                allContours.append([error, True, intersection])

    if len(allContours) == 0:
        allContours.append([0, False, intersection])

def detectSymbol(frame):
    global interpreter, input_details, output_details, symbol_names
    inputFrame = cv2.resize(frame, (224, 224))
    inputFrame = inputFrame.astype(np.float32) / 255.0
    inputFrame = np.expand_dims(inputFrame, axis=0)

    interpreter.set_tensor(input_details[0]['index'], inputFrame)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]
    predictedClass = np.argmax(output)
    confidence = output[predictedClass]

    if confidence > 0.5:
        label = f"{symbol_names[predictedClass]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return label
    return None

cam = setupCam()
rightPWM, leftPWM, servoPWM = setupGPIO()
interpreter, input_details, output_details, symbol_names = setupModel()
state = "NORMAL"
reverseStartTime = 0
currentScanIndex = 0
scanStartTime = 0
detectedScanAngle = None

"""
lowerRed1 = np.array([0, 80, 60])
upperRed1 = np.array([20, 255, 255])
lowerRed2 = np.array([160, 80, 60])
upperRed2 = np.array([180, 255, 255])
lowerBlue = np.array([100, 150, 60])
upperBlue = np.array([140, 255, 255])
lowerGreen = np.array([40, 70, 60])
upperGreen = np.array([80, 255, 255])
lowerYellow = np.array([20, 100, 60])
upperYellow = np.array([40, 255, 255])
lowerBlack = np.array([0, 0, 0])
upperBlack = np.array([180, 255, 120])
"""

lowerBlack = np.array([0, 0, 0])
upperBlack = np.array([180, 255, 120])
lowerRed1 = np.array([0, 150, 70])
upperRed1 = np.array([10, 255, 255])
lowerRed2 = np.array([160, 150, 70])
upperRed2 = np.array([180, 255, 255])
lowerYellow = np.array([20, 150, 150])
upperYellow = np.array([35, 255, 255])
lowerGreen = np.array([40, 70, 60])
upperGreen = np.array([80, 255, 255])
lowerBlue = np.array([100, 100, 70])
upperBlue = np.array([120, 255, 150])

#lowerGreen = np.array([60, 100, 100])
#upperGreen = np.array([90, 255, 200])
#lowerBlue = np.array([100, 150, 60])
#upperBlue = np.array([140, 255, 255])


try:
    rightPWM, leftPWM, servoPWM = setupGPIO()
    setServoAngleSimple(servoPWM, 90)
    interpreter, inputDetails, outputDetails, symbolNames = setupModel()
    cam = setupCam()
    priorityInput = input("Enter the colors that you want: ")
    priorityColors = []
    if 'b' in priorityInput:
        priorityColors.append("blue")
    if 'g' in priorityInput:
        priorityColors.append("green")
    if 'r' in priorityInput:
        priorityColors.append("red")
    if 'y' in priorityInput:
        priorityColors.append("yellow")
    
    priorityColors = list(set(priorityColors))[:4]
    print(f"Following colors: {priorityColors}")

    state = "NORMAL"
    reverseStartTime = 0
    currentScanIndex = 0
    scanStartTime = 0
    detectedScanAngle = None

    while True:
        frame = cam.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detectLine(frame)

        global allContours

        detectedSymbol = detectSymbol(frame_bgr)
        if detectedSymbol:
            print(f"Detected symbol: {detectedSymbol}")
            if detectedSymbol == "stop" or detectedSymbol == "handStop":
                state = "STOP"
                stopMotors(rightPWM, leftPWM)
                setServoAngleSimple(servoPWM, 90)
                print("STOP symbol detected. Stopping motors.")
                time.sleep(3)

        cv2.imshow('Line follower', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if state == "NORMAL":
            if allContours[0][1]:
                if allContours[0][0] > TURN_THRESHOLD:
                    pivotTurnLeft(rightPWM, leftPWM)
                    print("Turning right")
                elif allContours[0][0] < -TURN_THRESHOLD:
                    pivotTurnLeft(rightPWM, leftPWM)
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
                stopMotors(rightPWM, leftPWM)
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
    stopMotors(rightPWM, leftPWM)
    setServoAngleSimple(servoPWM, 90)
    gpio.cleanup()
    cam.close()
    cv2.destroyAllWindows()
    print("GPIO cleaned up and program terminated.")