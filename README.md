# UAV-Based-Disease-Identification-in-Solanum-lycopersicum-Leaves-Using-ML-and-DL-Algorithms
Overview

This project implements a Deep Learning and Machine Learning-based UAV-based tomato field health monitoring system using 3D CNN, 2D CNN, SVM, and Random Forest classifiers. The system analyzes UAV-captured RGB images and environmental sensor data (temperature, humidity, CO2 levels, and rainfall) to predict the health status of tomato plants (Healthy/Diseased).

Dataset

File: UAV_Tomato_Field_Dataset_Jan2025.csv

Attributes:

Date_Time: Timestamp of data collection

Temperature (°C): Normalized temperature values

Humidity (%): Normalized humidity values

Rainfall (mm): Normalized rainfall data

CO2 Levels (ppm): Normalized CO2 concentration

RGB Values: Tuple of (R, G, B) values from UAV images

Health Status: Binary label (0 = Healthy, 1 = Diseased)

Model Implementations

1. 3D CNN for Multimodal Data Fusion

Processes sensor data and RGB image data simultaneously.

Uses multiple Conv3D and MaxPooling3D layers.

Output: Binary classification (Healthy/Diseased).

2. 2D CNN for Image Classification

Trains a 2D CNN to classify plant health based on RGB values only.

Uses Conv2D layers followed by Dropout and BatchNormalization.

3. SVM & Random Forest for Sensor-Based Classification

Trains SVM and RF classifiers using only sensor data.

Compares traditional ML models with deep learning-based approaches.

Dependencies

Install the required Python libraries:

pip install numpy pandas tensorflow scikit-learn matplotlib seaborn

How to Run

Prepare the dataset: Ensure UAV_Tomato_Field_Dataset_Jan2025.csv is in the project directory.

Train and test models:

python tomotoML.py

Example Prediction:
Modify the sample sensor and RGB values in tomotoML.py to test with new data.

Results

The system evaluates model performance using accuracy, precision, recall, and F1-score.

A confusion matrix visualizes classification performance.

Models are compared using accuracy metrics.

Future Enhancements

Expand dataset with real UAV images for better generalization.

Hyperparameter tuning for improved CNN performance.

Integration with real-time UAV systems for live plant monitoring.

Author

-Dr.Lordwin Cecul Prabhaker
