# LVM Cam
Deploy a Large Vision Model (LVM) on your local machine and prompt it about frames from your USB camera. This application takes advantage of 4bit quantization and a GPU.

## Overview
Users can ask the app about the what is currently on the camera and the chatbot will answer questions and describe the image. 

## Requirements
1. Machine with GPU access
2. USB or built-in camera
3. 50 GB of storage ( 4bit quantized Llava is 15gb, but it seems like more space is used throughout the downloading/quantizing)

This demo was tested on a Windows 11 desktop with RTX 4080. It's possible to do this without utilizing a GPU, however, 4 bit quantization requires a GPU.