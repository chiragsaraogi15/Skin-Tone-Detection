# ToneSense: Discovering Diversity in Skin Tones through AI

ToneSense is an AI-powered project that aims to classify celebrity faces based on three different skin tones: Fair/Light, Medium/Tan, and Dark/Deep. It utilizes the CelebA dataset for training and evaluation, implements face detection using the MTCNN model, and performs skin tone classification using the MobileNetV2 model.

This repository contains two main Python notebooks and associated code for image processing and skin tone classification.

## Project Overview

ToneSense aims to explore the diversity in skin tones among celebrity faces using AI and machine learning techniques. The project consists of two main components:

1. **Image_Processing Notebook**: This notebook processes the original CelebA dataset. It extracts face images using the MTCNN face detection pre-trained model, combines them, and prepares the dataset for skin tone classification.

2. **Skin_Tone_Classification Notebook**: This notebook uses the processed face images to classify them based on three predefined skin tones: Fair/Light, Medium/Tan, and Dark/Deep. The MobileNetV2 model is employed for this classification task.

The project has been deployed on Streamlit, allowing users to interactively explore the results of the skin tone classification.

## Notebooks

1. **Image_Processing.ipynb**: This notebook covers the following steps:
   - Loading and preprocessing the CelebA dataset.
   - Face detection using the MTCNN pre-trained model.
   - Extracting and saving the detected face images.

2. **Skin_Tone_Classification.ipynb**: This notebook covers the following steps:
   - Loading the preprocessed face images.
   - Training the MobileNetV2 model for skin tone classification.
   - Evaluating the model's performance and saving it for deployment.


## Dataset

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
