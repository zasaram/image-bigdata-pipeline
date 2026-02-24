# Scalable Big Data Image Processing Pipeline using Transfer Learning and Deep Feature Extraction

## Overview
This project implements a scalable image processing pipeline for large-scale image classification using transfer learning.

A pretrained convolutional neural network (MobileNetV2) is used to extract deep features, and a custom classifier is trained for efficient and accurate image recognition.

## Key Features
- Large-scale image dataset handling
- Transfer learning with pretrained CNN
- Deep feature extraction
- Modular and configurable architecture
- Designed for Big Data image processing scenarios

## Project Structure
main.py – Training pipeline  
data_utils.py – Data loading and preprocessing  
model_utils.py – Model architecture  
config_example.py – Configuration template  

## Dataset Structure
data/
    class1/
    class2/
    class3/

## Usage
1. Copy config_example.py as config.py and modify paths if needed
2. Place your dataset inside the data folder
3. Run:

python main.py

## Applications
- Large-scale image classification
- Visual data analytics
- Feature extraction for machine learning
- Computer vision pipelines