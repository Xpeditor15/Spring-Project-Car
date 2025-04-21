import time
import cv2
import numpy as np
import RPi.GPIO as gpio
from picamera2 import Picamera2

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

    LEFT_PWM_FWD = gpio.PWM(IN2, 100)
    pwmEnb = gpio.PWM(ENB, 100)
    pwmServo = gpio.PWM(SERVO, 50)

    pwmEna.start(0)
    pwmEnb.start(0)
    pwmServo.start(0)

    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.HIGH)
    
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
        