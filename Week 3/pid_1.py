import time
import cv2
import numpy as np
import RPi.GPIO as gpio
from picamera2 import Picamera2

#Using PID, detects black line only, uses software PWM

IN1, IN2, IN3, IN4, ENA, ENB, SERVO = 23, 24, 27, 22, 12, 13, 26
MAX_SPEED = 60
MIN_SPEED = 20
TURN_SPEED = 50
TURN_THRESHOLD = 40

class PID:
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

def setup():
    gpio.setmode(gpio.BCM)

    gpio.setup([IN1, IN2, IN3, IN4, ENA, ENB, SERVO], gpio.OUT)

    global LEFT_PWM_FWD
    global LEFT_PWM_BWD
    global RIGHT_PWM_FWD
    global RIGHT_PWM_BWD
    global pwmServo

    LEFT_PWM_FWD = gpio.PWM(IN2, 100)
    LEFT_PWM_BWD = gpio.PWM(IN1, 100)
    RIGHT_PWM_FWD = gpio.PWM(IN4, 100)
    RIGHT_PWM_BWD = gpio.PWM(IN3, 100)
    pwmServo = gpio.PWM(SERVO, 50)

    LEFT_PWM_FWD.start(0)
    LEFT_PWM_BWD.start(0)
    RIGHT_PWM_FWD.start(0)
    RIGHT_PWM_BWD.start(0)
    pwmServo.start(0)
    
def camSetup():
    global camera
    camera = Picamera2()
    camera.resolution = (320, 240)
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
    camera.framerate = 30
    camera.start()

def preprocessFrame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 75])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    return mask, hsv

def findLine(thresh):
    roi = thresh[120:280, :]
    contours, _ = cv2.findContours(roi, cv2.RETR_REE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largestContours = max(contours, key=cv2.contourArea)
        M = cv2.moments(largestContours)

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

    leftSpeed = MAX_SPEED - correction
    rightSpeed =MAX_SPEED + correction

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

        if cv2.waitKey(1) & 0xFFF == ord('q'):
            break

finally:
    cv2.destroyAllwindows()
    LEFT_PWM_FWD.stop()
    LEFT_PWM_BWD.stop()
    RIGHT_PWM_FWD.stop()
    RIGHT_PWM_BWD.stop()
    gpio.cleanup()
    camera.close()