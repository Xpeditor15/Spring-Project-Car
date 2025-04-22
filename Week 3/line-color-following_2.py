import RPi.GPIO as gpio
import time
import numpy as np
import cv2
from picamera2 import Picamera2

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

BASE_SPEED = 50
TURN_SPEED = 65
MIN_CONTOUR_AREA = 250
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

TURN_THRESHOLD = 60

REVERSE_DURATION = 0.3
REVERSE_SPEED = 40

SCAN_ANGLES = [90, 45, 135]
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
        print(f"Detected angle {scannedAngle}, pivoting left")
        gpio.output(IN1, gpio.LOW)
        gpio.output(IN2, gpio.HIGH)
        gpio.output(IN3, gpio.HIGH)
        gpio.output(IN4, gpio.LOW)
        rightPWM.ChangeDutyCycle(TURN_SPEED)
        leftPWM.ChangeDutyCycle(TURN_SPEED)
    elif scannedAngle < 90:
        print(f"Detected angle {scannedAngle}, pivoting right")
        gpio.output(IN1, gpio.HIGH)
        gpio.output(IN2, gpio.LOW)
        gpio.output(IN3, gpio.LOW)
        gpio.output(IN4, gpio.HIGH)
        rightPWM.ChangeDutyCycle(TURN_SPEED)
        leftPWM.ChangeDutyCycle(TURN_SPEED)
    else:
        print("Detected angle 90: no pivot required")
        return

    