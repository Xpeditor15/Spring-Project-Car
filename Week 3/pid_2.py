import time
import cv2
import numpy as np
import RPi.GPIO as gpio
from picamera2 import Picamera2

#This code only follows black line using PID and uses the hardware PWM pins instead of software

LEFT_MOTOR_FORWARD = 12
LEFT_MOTOR_BACKWARD = 13
RIGHT_MOTOR_FORWARD = 18
RIGHT_MOTOR_BACKWARD = 19

gpio.setmode(gpio.BCM)
gpio.setup([LEFT_MOTOR_FORWARD, LEFT_MOTOR_BACKWARD, RIGHT_MOTOR_FORWARD, RIGHT_MOTOR_BACKWARD], gpio.out)

LEFT_PWM_FWD = gpio.PWM(LEFT_MOTOR_FORWARD, 1000)
LEFT_PWM_BWD = gpio.PWM(LEFT_MOTOR_BACKWARD, 1000)
RIGHT_PWM_FWD = gpio.PWM(RIGHT_MOTOR_FORWARD, 1000)
RIGHT_PWM_BWD = gpio.PWM(RIGHT_MOTOR_BACKWARD, 1000)

LEFT_PWM_FWD.start(0)
LEFT_PWM_BWD.start(0)
RIGHT_PWM_FWD.start(0)
RIGHT_PWM_BWD.start(0)

MAX_SPEED = 80
MIN_SPEED = 20
TURN_SPEED = 50
TURN_THRESHOLD = 40

class PID():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previousError = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.previousError
        self.previousError = error
        correction = self.kp * error + self.ki * self.integral + self.kd * derivative
        return correction

pid = PID(1, 0.01, 0.1)

def camSetup():
    global camera

    camera = Picamera2()
    camera.preview_configuration.main.size = (320, 240)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.controls.Framerate = 30
    camera.configure("preview")
    camera.start()
    time.sleep(1)

def preprocessFrame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lowerBlack = np.array([0, 0, 0])
    upperBlack = np.array([180, 255, 100])
    mask = cv2.inRange(hsv, lowerBlack, upperBlack)
    return mask, hsv

def findLine(thresh):
    roi = thresh[120:240, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largestContour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largestContour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            return cx
    
    return None

def controlMotors(cx, frameWidth):
    centerX = frameWidth // 2

    if cx is None:
        LEFT_PWM_FWD.ChangeDutyCycle(0)
        LEFT_PWM_BWD.ChangeDutyCycle(0)
        RIGHT_PWM_FWD.ChangeDutyCycle(0)
        RIGHT_PWM_BWD.ChangeDutyCycle(0)
        return
    
    error = cx - centerX

    if abs(error) > TURN_THRESHOLD:
        if error > 0:
            LEFT_PWM_FWD.ChangeDutyCycle(0)
            LEFT_PWM_BWD.ChangeDutyCycle(TURN_SPEED)
            RIGHT_PWM_FWD.ChangeDutyCycle(TURN_SPEED)
            RIGHT_PWM_BWD.ChangeDutyCycle(0)
        else:
            LEFT_PWM_FWD.ChangeDutyCycle(TURN_SPEED)
            LEFT_PWM_BWD.ChangeDutyCycle(0)
            RIGHT_PWM_FWD.ChangeDutyCycle(0)
            RIGHT_PWM_BWD.ChangeDutyCycle(TURN_SPEED)
        return
    
    correction = pid.compute(error)

    leftSpeed = max(min(leftSpeed, MAX_SPEED), MIN_SPEED)
    rightSpeed = max(min(rightSpeed, MAX_SPEED), MIN_SPEED)

    LEFT_PWM_FWD.ChangeDutyCycle(leftSpeed)
    LEFT_PWM_BWD.ChangeDutyCycle(0)
    RIGHT_PWM_FWD.ChangeDutyCycle(rightSpeed)
    RIGHT_PWM_BWD.ChangeDutyCycle(0)

    print(f"Error: {error}, Correction: {correction}, Left Speed: {leftSpeed}, Right Speed: {rightSpeed}")

try:
    while True:
        frame = camera.capture_array()

        mask, hsv = preprocessFrame(frame)

        cx = findLine(mask)

        controlMotors(cx, frame.shape[1])

        cv2.imshow("RGB Preview", frame)
        cv2.imshow("HSV Preview", hsv)
        cv2.imshow("Thresholded Image", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cv2.destroyAllWindows()
    LEFT_PWM_FWD.stop()
    LEFT_PWM_BWD.stop()
    RIGHT_PWM_FWD.stop()
    RIGHT_PWM_BWD.stop()
    gpio.cleanup()
    camera.close()