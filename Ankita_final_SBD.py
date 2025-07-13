import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np

# Setup GPIO in BCM mode
GPIO.setmode(GPIO.BCM)

# Motor setup
Motor1A = 23
Motor1B = 24
Motor1Enable = 12
Motor2A = 25
Motor2B = 26
Motor2Enable = 13
Motor3A = 5
Motor3B = 6
Motor3Enable = 18
Motor4A = 16
Motor4B = 20
Motor4Enable = 19

motor_pins = [Motor1A, Motor1B, Motor2A, Motor2B, Motor3A, Motor3B, Motor4A, Motor4B]
enable_pins = [Motor1Enable, Motor2Enable, Motor3Enable, Motor4Enable]
GPIO.setup(motor_pins + enable_pins, GPIO.OUT)

pwms = [
    GPIO.PWM(Motor1Enable, 1000),
    GPIO.PWM(Motor2Enable, 1000),
    GPIO.PWM(Motor3Enable, 1000),
    GPIO.PWM(Motor4Enable, 1000)
]

for pwm in pwms:
    pwm.start(100)  # Start motors at full speed initially

# Camera and model setup
model = YOLO("/home/pi/best-5.pt")
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# Buzzer setup
BUZZER_PIN = 17
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def start_buzzer():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)

def stop_buzzer():
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def run_motors_forward(speed):
    for pwm in pwms:
        pwm.ChangeDutyCycle(speed)

def slow_down_motors():
    for duty_cycle in range(100, 0, -10):
        for pwm in pwms:
            pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.2)

def speed_up_motors():
    for duty_cycle in range(0, 101, 10):  # Gradually increase speed from 0% to 100%
        for pwm in pwms:
            pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.2)

frame_skip = 5
frame_count = 0
detected = False
detection_start_time = None
buzzer_active = False

try:
    while True:
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        frame_count += 1

        if frame_count % frame_skip == 0:
            results = model.predict(frame_rgb, conf=0.4)  # Apply a 0.4 confidence threshold directly
            new_detection = False

            for box in results[0].boxes:
                if box.conf[0] > 0.4:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    label = f"speed-bump {conf:.2f}"
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    new_detection = True

            if new_detection:
                if not detected:
                    detected = True
                    detection_start_time = time.time()
                    start_buzzer()
                    slow_down_motors()
            else:
                if detected:
                    detected = False
                    detection_start_time = None
                    stop_buzzer()
                    speed_up_motors()  # Gradually increase speed to normal

        # Display frame for debugging
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow("YOLOv8 Detection", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    for pwm in pwms:
        pwm.stop()
    picam2.stop()
    stop_buzzer()
    GPIO.cleanup()
    cv2.destroyAllWindows()

