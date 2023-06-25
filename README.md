# Handwritten Digit Recognition Model
## Introduction
A Handwritten Digit Recognition Model trained using MNIST Dataset can detect the digit in scanned form(png) or using Python GUI. 
## Dataset
 [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) - 
 The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in machine learning.
## Required Modules/Libraries
- Keras
- TensorFlow
- NumPy
- matplotlib
- OpenCV (cv2)
- pygame
- sys
## Files Description
#### Digit_Recognition.ipynb:
It is the main file of our application that creates the model for detecting the handwritten digit. We will save the model in .h5 form to import/load the model into the test file.

#### model.h5
We import our model to the app.py/test.py to detect the handwritten image. Note that it should be in the same directory as the app.py/test.py.

#### test.ipynb:
It is the test file of our application that takes an input image of a handwritten digit and detects it with the help of our model.

#### app.py
It is a GUI application built in Python using the Pygame module. It provides an Interface to the user for writing the digit. It is also used for testing the accuracy of our model and thereby predicting the digit by showing its label above & draws a bounding box around the digit.<br>
**Note**: We can clear the screen if it is full by pressing Backspace key in keyboard.
## Accuracy
The accuracy of our model is around 99 %. It can also be improved slightly by changing the model parameters, such as increasing the no of hidden layers (or), changing the value of the Dropout layer (or) batch size, etc. 
