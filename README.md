# Lung_Disease_Detection
Deep Learning Lung Disease Detection Project

## Overview
This project involves building a deep learning model for image classification using the ResNet50 architecture. The dataset used for training and testing the model consists of images categorized into different classes. The main goal is to predict the class labels of images accurately.

## Project Structure
- `data`: Contains the dataset used for training and testing.
- `model_resnet50.h5`: The trained ResNet50 model saved in HDF5 format.
- `README.txt`: Documentation providing an overview of the project, instructions, and details about the code and model.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Pandas
- Scikit-learn

Dowload reqired libraries just by running following command:
```
pip install tensorflow numpy opencv-contrib-python opencv-python matplotlib seaborn pandas scikit-learn
```

## File Descriptions
- `image_classification.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and prediction.
- `README.txt`: This file, providing essential information about the project.

## Instructions
1. *Environment Setup:*
   - Install the required Python packages listed in `requirements.txt` by running:
     
     pip install -r requirements.txt
     

2. *Dataset:*
   - Place the image dataset in the `data` directory.

3. *Model Training:*
   - Run the `image_classification.ipynb` notebook to train the ResNet50 model. Adjust hyperparameters and paths as needed.

4. *Model Evaluation:*
   - Evaluate the model on the test set and view performance metrics using the provided code in the notebook.

5. *Prediction:*
   - Use the model to make predictions on new images by following the example provided in the notebook.

6. *Results Visualization:*
   - Visualize model predictions on a subset of the test set using Matplotlib.

## Additional Notes
- The trained model (`model_resnet50.h5`) can be loaded for further use without retraining.
- Adjust the image paths and parameters in the notebook according to your project structure and requirements.

## Running Locally
Navigate to the project directory containing Projectrun.py and run the following command in the terminal:
```
python Projectrun.py
```

Feel free to customize and expand upon this README to suit your project's specific details and requirements.
