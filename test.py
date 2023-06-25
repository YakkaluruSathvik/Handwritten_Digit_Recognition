import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model

MODEL = load_model("model.h5")
image_no = 9

# input_img = cv2.imread('sample_images/image_{}.png'.format(image_no))
input_img = cv2.imread('test_images/{}.png'.format(image_no))

print(input_img)
print(input_img.shape)

gray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
print(gray.shape)

image_resize = cv2.resize(gray,(28,28))
print(image_resize.shape)

image_resize = image_resize/255
image_reshape = np.reshape(image_resize,[1,28,28])

prediction = MODEL.predict(image_reshape)
print(f"The Predicted digit is {np.argmax(prediction)}")