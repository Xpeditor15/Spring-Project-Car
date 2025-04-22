#uses PID for movement, has issues with pid values as car is oscillating very quickly

import cv2
import numpy as np
import RPi.GPIO as gpio
import time 
from picamera2 import Picamera2
import threading

IN1, IN2, IN3, IN4, ENA, ENB, SERVO= 23, 24, 27, 22, 12, 13, 26 #Motor pins
hasReverse = False
canContinueMovement = False


MIN_CONTOUR_AREA = 200
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
ROI_HEIGHT = 80
ROI_OFFSET = 200

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
        derivative = (error - self.previousError) / timeDelta
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previousError = error
        self.previousTime = currentTime
        return output
    
def setMotorSpeed(motor, speed): #Function to easily change the speed of the left and right motor, automatically decides the direction
    if motor == 'left':
        pin1, pin2 = IN1, IN2
    elif motor == 'right':
        pin1, pin2 = IN3, IN4
        
    if speed > 0:
        gpio.output(pin1, gpio.LOW)
        gpio.output(pin2, gpio.HIGH)
    elif speed < 0: 
        gpio.output(pin1, gpio.HIGH)
        gpio.output(pin2, gpio.LOW)
    duty = min(max(abs(speed) * 8, 50), 60)
    print(f"duty: {duty}")
    if speed == 0:
        duty = 0
    
    if motor == 'left':
        pwmEnb.ChangeDutyCycle(duty)
    elif motor == 'right':
        pwmEna.ChangeDutyCycle(duty)

def reverse():
    gpio.output(IN1, gpio.HIGH)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.HIGH)
    gpio.output(IN4, gpio.LOW)

    pwmEna.ChangeDutyCycle(50)
    pwmEnb.ChangeDutyCycle(50)

    time.sleep(0.5)

    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.LOW)

def setAngle(angle):
    duty = (angle / 18) + 2.5
    pwmServo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwmServo.ChangeDutyCycle(0)

def searchLine(frame):
    setMotorSpeed('left', 0)
    setMotorSpeed('right', 0)

    reverse()
    hasReverse = True

    setAngle(0)

    while True:
        for angle in range (0, 181, 10):
            setAngle(angle)
            output = detectLine(frame)
            if output != 0:
                break
        break

    canContinueMovement = True

    time.sleep(5)
    
    setAngle(90) 
    hasReverse = False


def detectLine(frame): #Function used to detect the line and returns the correction value for PID
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 75])
    
    roi = frame[FRAME_HEIGHT-ROI_OFFSET: FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0: FRAME_WIDTH]
    roi_hsv = hsv[FRAME_HEIGHT-ROI_OFFSET: FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT, 0: FRAME_WIDTH]

    mask_black = cv2.inRange(roi_hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.rectangle(frame, (0, FRAME_HEIGHT-ROI_OFFSET), (FRAME_WIDTH, FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT), (0, 255, 0), 2)

    center_x = FRAME_WIDTH // 2
    cv2.line(frame, (center_x, FRAME_HEIGHT-ROI_OFFSET), (center_x, FRAME_HEIGHT-ROI_OFFSET+ROI_HEIGHT), (0, 0, 255), 2)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                cv2.circle(roi, (cx, cy), 5, (255, 0, 0), -1)

                error = cx - center_x
                
                if abs(error) < 20:
                    error = 0

                cv2.line(roi, (center_x, cy), (cx, cy), (255, 0, 0), 2)

                return pid.compute(error)
    else:
        return 0    

pid = PID(kp=0.14, ki=0, kd=0.005)
base_speed = 0.1
lastLine = 160

setup()

setupCam()

try:
    while True:
        frame = camera.capture_array()

        correction = detectLine(frame)

        if correction == 0: #activated when straight line, wrong logic
            if (hasReverse is False):
                reverse()
            thread = threading.Thread(target=searchLine)
            thread.start()
            
        print(f"correction: {correction}")

        rightSpeed = base_speed + correction
        leftSpeed = base_speed - correction

        setMotorSpeed('left', leftSpeed)
        setMotorSpeed('right', rightSpeed)

        cv2.putText(frame, f"Left: {leftSpeed:.1f} | Right: {rightSpeed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("CAR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("stopped by user")
    gpio.output(IN1, gpio.LOW)
    gpio.output(IN2, gpio.LOW)
    gpio.output(IN3, gpio.LOW)
    gpio.output(IN4, gpio.LOW)
    pwmEna.ChangeDutyCycle(0)
    pwmEna.ChangeDutyCycle(0)
    gpio.cleanup()

