# SmartRoadSafety-YOLOv8
Real-time speed breaker detection and adaptive braking system using YOLOv8 and Raspberry Pi4 for safer road navigation.

# YOLOv8-Powered Speed Breaker Detection and Adaptive Braking

This project demonstrates a real-time embedded system that detects speed breakers using the YOLOv8 object detection model and applies adaptive braking using a Raspberry Pi. It aims to enhance road safety, especially for autonomous and electric vehicles.

## Project Overview

- Title: YOLOv8-Powered Speed Breaker Detection and Adaptive Braking for Safer Road Navigation
- Developed by: Ankita Gupta  
- Institute: Symbiosis Institute of Technology, Pune  
- Guided by: Dr. Shripad Deshpande  
- Industry Mentors: Mr. Mukul Malviya, Mr. Om Prakash Bharthuar (JSW MG Motors)

## Key Features

- Object detection using YOLOv8
- Real-time image capture using Raspberry Pi Camera
- Buzzer and motor response on speed bump detection
- Adaptive speed control using PWM and L298N motor driver
- Deployment and testing on Raspberry Pi 4

## Dataset

- ~2000 images of marked and unmarked speed breakers
- Sources: Mendeley, Kaggle, and open datasets
- Annotated using Roboflow (polygon tool)

## Hardware Used

- Raspberry Pi 4
- PiCamera V2
- L298N Motor Driver
- DC Motors with toy car chassis
- Buzzer for alert
- Google Coral USB Accelerator (optional for FPS improvement)

## Software Stack

- YOLOv8 (Ultralytics)
- Python (OpenCV, Picamera2, RPi.GPIO, gpiozero)
- Roboflow for annotation and augmentation
- Google Colab for model training
- Thonny IDE for deployment scripts

## Limitations

- Inference delay on Raspberry Pi (~1.5 seconds per frame)
- Small dataset and limited lighting conditions

## Future Scope

- Switch to Jetson Nano for better FPS
- Dataset expansion to 10,000+ images
- Integrate with real vehicle via CAN bus
- Improve low-light detection and adaptive braking accuracy


