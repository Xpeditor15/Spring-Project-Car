import cv2
import numpy as np
import RPi.gpio as gpio
import time
from picamera2 import Picamera2

IN1, IN2, IN3, IN4, ENA, ENB, SERVO = 23, 24, 27, 22, 12, 13, 26
previousTime = time.time()

MIN_CONTOUR_AREA = 200
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
ROI_HEIGHT = 80
ROI_OFFSET = 100

def setup(): #Configures the setup of the motor within the function
    gpio.setmode(gpio.BCM)

    gpio.setup(ENA, gpio.OUT)
    gpio.setup(IN1, gpio.OUT)
    gpio.setup(IN2, gpio.OUT)
    gpio.setup(IN3, gpio.OUT)
    gpio.setup(IN4, gpio.OUT)
    gpio.setup(ENB, gpio.OUT)
    gpio.setup(26, gpio.OUT)

    global pwmEna
    global pwmEnb
    global pwmServo
    pwmEna = gpio.PWM(ENA, 1000)
    pwmEnb = gpio.PWM(ENB, 1000)
    pwmServo = gpio.PWM(SERVO, 50)
    pwmEna.start(0)
    pwmEnb.start(0)
    pwmServo.start(0)

    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.HIGH)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.HIGH)

def setupCam(): #Configures the setup of the camera within the function
    global camera
    camera = Picamera2()
    camera.resolution = (320, 240)
    camera.configure(camera.create_preview_configuration(main={"format": "RGB888", "size": (320, 240)}))
    camera.framerate = 30
    camera.start()

class PID: #Defines the PID class, contains the compute function
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previousError = 0
        self.previousTime = time.time()

    def compute(self, error):
        currentTime = time.time()
        timeDelta = currentTime - self.previousTime
        if timeDelta == 0:
            timeDelta = 1e-6
        self.integral += error * timeDelta
        if self.integral > 10: #resets the integral if its too big
            self.integral = 0 
        derivative = (error - self.previousError) / timeDelta
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previousError = error
        self.previousTime = currentTime
        return output

def reverse():
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)

    pwmEna.ChangeDutyCycle(45)
    pwmEna.ChangeDutyCycle(45)

def setMotorSpeed(motor, speed): 
    if motor == 'left':
        pin1, pin2 = IN1, IN2
    elif motor == 'right':
        pin1, pin2 = IN3, IN4

