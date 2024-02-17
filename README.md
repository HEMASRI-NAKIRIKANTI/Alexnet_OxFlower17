# Oxford Flower 17 Dataset - CNN Classification

## Introduction

This is a simple Convolutional Neural Network (CNN) classification project using the Oxford Flower 17 dataset. The model is built using the Keras library with TensorFlow backend.

## Dataset

The dataset used in this project is the Oxford Flower 17 dataset. It contains 1360 images of flowers from 17 different categories. The images are in the form of RGB images with a resolution of 224x224 pixels.

The dataset can be downloaded from Kaggle at the following link: [Oxford Flower 17 Dataset](https://www.kaggle.com/datasets/haesunpark/oxflower17?resource=download)

## Model Architecture

The model architecture used in this project is based on the AlexNet architecture. It consists of five convolutional layers followed by max-pooling, batch normalization, and dropout layers. The last layer is a softmax activation layer for multi-class classification.

## Results

The model was trained on 1088 samples and validated on 272 samples. After 1 epoch of training, the training accuracy was approximately 25%, and the validation accuracy was approximately 5.51%. These accuracies are quite low, indicating that the model needs further refinement and optimization to achieve better accuracy.

## Usage

To use this code, you need to have the following libraries installed:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

You can install these libraries using pip:

