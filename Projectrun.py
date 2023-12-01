import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the saved ResNet50 model
model = load_model("model_resnet50.h5")

# Category List
category = list(os.listdir("E:/Study/Degree/Sem 7/Deep Learning (01CT0722)/Project/dataset"))

# Image path
img_path = "E:/Study/Degree/Sem 7/Deep Learning (01CT0722)/Project/Test data/00028173_005.png"

# Read and preprocess the image
img = cv2.imread(img_path)
img = cv2.resize(img, (100, 100))
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Make predictions using the loaded model
result = model.predict(x)

# Print the results
print("Predictions:")
print(result)
print("Predicted Class:", category[np.argmax(result)])