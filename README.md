# Highway Vehicle Counting Practice Exercise

## Overview

This repository contains a Jupyter notebook demonstrating a highway vehicle counting system using computer vision techniques. The project showcases the application of object detection and clustering algorithms to analyze traffic patterns in highway footage.

## Project Description

The core of this project involves using the YOLO (You Only Look Once) object detection model to identify vehicles in an image of a highway. After detection, the script applies K-means clustering to group the vehicles based on their positions.

Key features of the project include:

1. Vehicle Detection: Utilizes the YOLO model to detect and locate vehicles in the image.
2. Clustering: Applies K-means clustering to group detected vehicles, potentially identifying lanes or traffic patterns.
3. Visualization: Draws bounding boxes around detected vehicles and marks cluster centers, providing a clear visual representation of the detection and clustering results.
4. Vehicle Counting: Counts the number of vehicles in each identified cluster, offering insights into traffic distribution.

This exercise serves as a foundation for more complex traffic analysis systems, demonstrating how computer vision and machine learning techniques can be applied to real-world traffic monitoring scenarios.

The project is implemented in Python, leveraging libraries such as OpenCV for image processing, Ultralytics for YOLO implementation, and scikit-learn for K-means clustering.
