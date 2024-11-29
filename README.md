# Facial Emotion Detection using YOLOv8

This repository contains code for facial emotion detection using a custom-trained YOLOv8 model. The model is trained to recognize various facial emotions from a dataset sourced from [Roboflow](https://universe.roboflow.com/laschanh/facial_emotion_detection/dataset/2). The emotions include **Angry**, **Disgusted**, **Fearful**, **Happy**, **Neutral**, **Sad**, and **Surprised**.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Setup and Installation](#setup-and-installation)
4. [Training the Model](#training-the-model)
5. [Running the Detection](#running-the-detection)
6. [License](#license)

## Introduction

This project leverages the YOLOv8 model for custom object detection to recognize facial emotions. The model was trained using a dataset from Roboflow, which contains images labeled with various emotional states.

The main components of the repository are:
- **YOLOv8 model**: Used for emotion detection in facial images.
- **Custom dataset**: A dataset containing seven emotion labels for facial expressions.
- **Detection code**: A Python script to apply the trained YOLOv8 model for real-time emotion recognition from webcam input.

## Dataset

The dataset used in this project was sourced from Roboflow and contains images of faces with various emotional expressions. Below is the dataset configuration:

### Dataset Information

- **Emotion Categories**:
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised
- **Number of Classes (nc)**: 7

### Dataset Split

- **Training set**: Contains images used to train the model.
- **Validation set**: Contains images used to validate the model during training.
- **Test set**: Contains images for testing the trained model.

### Dataset YAML File

```yaml
names:
  - Angry
  - Disgusted
  - Fearful
  - Happy
  - Neutral
  - Sad
  - Surprised
nc: 7
roboflow:
  license: CC BY 4.0
  project: facial_emotion_detection
  url: https://universe.roboflow.com/laschanh/facial_emotion_detection/dataset/2
  version: 2
  workspace: laschanh
test: /home/omar_ben_emad/State of The Art/Facial_Emotion/Facial_Emotion_Detection-2/test/images
train: /home/omar_ben_emad/State of The Art/Facial_Emotion/Facial_Emotion_Detection-2/train/images
val: /home/omar_ben_emad/State of The Art/Facial_Emotion/Facial_Emotion_Detection-2/valid/images
```

## Training the Model

To train the YOLOv8 model with your custom dataset, follow these steps:

1. **Prepare Dataset**:

   Make sure your dataset is properly organized as per the structure mentioned in the `dataset.yaml` file (train, val, and test directories).

2. **Train the Model**:

   Run the following command to start training the model. Make sure that the paths in your `dataset.yaml` are correct.

   ```bash
   yolo train model=yolov8n.yaml data=path/to/dataset.yaml epochs=100 batch=16 imgsz=640
    ```

## Running the Detection
To run the emotion detection on a webcam feed, use the following Python script:

   ```bash
python detect_emotion.py
```
This script loads the YOLOv8 model and runs the webcam feed, detecting emotions in real-time.
