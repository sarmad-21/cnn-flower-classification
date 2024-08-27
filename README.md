# Flower Image Classification with CNN 

## Overview 
This project focuses on classifying flower images into two categories: "0" and "5". The model uses Convolutional Neural Networks (CNNs) to achieve high classification performance. The primary goal is to accurately categorize flower images, leveraging deep learning techniques.

## Technologies Used
- Python
- TensorFlow
- Keras
- PIL (Python Imaging Library)
- NumPy
- Scikit-Learn
- OS

## Deep Learning Model 
The model uses a Convolutional Neural Network (CNN) architecture to classify images. Key components include:
- **Convolutional Layer**: Extracts features from the input images. In this model, the convolutional layer uses 16 filters to capture different features from the image.
- **ReLU Activation Function**: Introduces non-linearity to the model.
- **Batch Normalization**: Normalizes the output of the previous layer to improve training speed and stability.
- **Global Average Pooling Layer**: Reduces the dimensionality of the feature maps by averaging across the spatial dimensions.
- **Dense Layer**: Connects the output of the pooling layer to the final output layer.
- **Sigmoid Activation Function**: Produces a probability score for binary classification.

## Data 
- Link to data: https://web.njit.edu/~usman/courses/cs677_spring21/flowers-recognition.zip
- The dataset consists of images of flowers and their corresponding labels.
- Labels are provided in the flower_labels.csv file.
- Images are resized to 32x32 pixels for processing.
- The data is split into training and validation sets using a 80/20 split.

## Data Augmentation 
The `ImageDataGenerator` class from Keras is used for real-time data augmentation during training. This helps in increasing the diversity of the training dataset and can lead to better generalization of the model. The augmentation parameters include:
- **Rotation Range**: 20 degrees
- **Width Shift Range**: 0.2
- **Height Shift Range**: 0.2
- **Shear Range**: 0.2
- **Zoom Range**: 0.2
- **Horizontal Flip**: Enabled
- **Fill Mode**: 'nearest'

## Model Training 
The model is trained using the following parameters:
- **Batch Size**: 2
- **Epochs**: 50
- **Loss Function**: Binary Crossentropy
- **ReduceLROnPlateau**: Adjusts the learning rate during training when the validation loss plateaus. It helps in improving model convergence by reducing the learning rate when performance stops improving.
- **EarlyStopping**: Stops training when the validation loss does not improve for a specified number of epochs. It helps in preventing overfitting.

## Performance 
The validation accuracy reached 80% which is greater compared to the MLP model's validation accuracy of 70%.

