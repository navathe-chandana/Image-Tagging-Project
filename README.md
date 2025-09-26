# Image Tagging Project 🖼️

A simple **Convolutional Neural Network (CNN)** built using **TensorFlow** to classify images from the **CIFAR-10 dataset**.  
This project is part of my **ShadowFox AIML Internship** tasks.

---

## 🔹 Project Description

The project aims to develop a practical solution for **image classification** by training a model to recognize 10 basic classes:

- airplane  
- car  
- bird  
- cat  
- deer  
- dog  
- frog  
- horse  
- ship  
- truck  

The model is trained on **50,000 training images** and tested on **10,000 images**.  

---

## 🔹 Features

- Preprocessing and normalization of images.  
- CNN architecture with 3 convolutional layers and 2 max-pooling layers.  
- Model training and evaluation on CIFAR-10.  
- Visualizations:
  - Sample images from dataset (`samples.png`)
  - Model predictions on test samples (`prediction_0.png` … `prediction_4.png`)
  - Training accuracy and loss graph (`history.png`)
- Trained model saved as `image_classifier.h5`.  

---

#Image Tagging Project Outputs 🖼️

This folder contains screenshots showing the **training progress, model evaluation, and predictions** of the Image Tagging Project using CNN and CIFAR-10 dataset.

---

## 🔹 Training Progress

These screenshots show the **model training across epochs**, including accuracy and loss improvements over time.

![Training Epochs 1](screenshots/training_epoch_1.png)

<img width="1920" height="1080" alt="Screenshot 2025-09-26 090301" src="https://github.com/user-attachments/assets/cf045325-6e5f-442c-bd42-0ed4854ec7e3" />

# 🔹 Accuracy, and Predictions
<img width="1920" height="1080" alt="Screenshot 2025-09-26 090321" src="https://github.com/user-attachments/assets/98f53a61-9419-4cec-bf83-c786f0729e9a" />



## 🔹 Requirements

Make sure you have Python installed (preferably **Python 3.10+**) and the following libraries:

```bash
pip install tensorflow numpy matplotlib pillow




