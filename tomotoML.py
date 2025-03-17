#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data for 3D CNN
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)  # Shape: (samples, 1, 1, features, 1)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    MaxPooling3D(pool_size=(1,1,1)),
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    MaxPooling3D(pool_size=(1,1,1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict health status for new data
def predict_health_status(sensor_data, image_rgb):
    sensor_data = np.array(sensor_data).reshape(1, 1, 1, 4, 1)  # Reshape for model input
    image_rgb = np.array(image_rgb).reshape(1, 1, 1, 3, 1)
    input_data = np.concatenate((image_rgb, sensor_data), axis=3)
    prediction = model.predict(input_data)
    return "Healthy" if prediction < 0.5 else "Diseased"

# Example prediction
sample_sensor_data = [30/40.0, 70/100.0, 5/20.0, 400/1000.0]  # Normalized sample
sample_rgb = [120, 200, 80]  # Sample RGB value
print("Predicted Health Status:", predict_health_status(sample_sensor_data, sample_rgb))


# In[2]:


pip install tensorflow


# In[3]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate timestamps for every 30 minutes in January 2025
start_date = datetime(2025, 1, 1, 0, 0)
end_date = datetime(2025, 1, 30, 23, 30)
timestamps = pd.date_range(start=start_date, end=end_date, freq='30min')

# Generate synthetic sensor values
np.random.seed(42)  # For reproducibility
temperature = np.random.uniform(20, 35, len(timestamps))  # Temperature in °C
humidity = np.random.uniform(60, 90, len(timestamps))  # Humidity in %
rainfall = np.random.uniform(0, 10, len(timestamps))  # Rainfall in mm
co2_levels = np.random.uniform(300, 600, len(timestamps))  # CO2 levels in ppm

# Generate random RGB values for UAV-captured images
rgb_values = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(len(timestamps))]

# Create DataFrame
dataset = pd.DataFrame({
    "Date_Time": timestamps,
    "Temperature (°C)": temperature,
    "Humidity (%)": humidity,
    "Rainfall (mm)": rainfall,
    "CO2 Levels (ppm)": co2_levels,
    "RGB Values": rgb_values
})

# Save dataset to CSV
dataset.to_csv("UAV_Tomato_Field_Dataset_Jan2025.csv", index=False)

print("Dataset generated and saved as UAV_Tomato_Field_Dataset_Jan2025.csv")


# In[4]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data for 3D CNN
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)  # Shape: (samples, 1, 1, features, 1)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    MaxPooling3D(pool_size=(1,1,1)),
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    MaxPooling3D(pool_size=(1,1,1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict health status for new data
def predict_health_status(sensor_data, image_rgb):
    sensor_data = np.array(sensor_data).reshape(1, 1, 1, 4, 1)  # Reshape for model input
    image_rgb = np.array(image_rgb).reshape(1, 1, 1, 3, 1)
    input_data = np.concatenate((image_rgb, sensor_data), axis=3)
    prediction = model.predict(input_data)
    return "Healthy" if prediction < 0.5 else "Diseased"

# Example prediction
sample_sensor_data = [30/40.0, 70/100.0, 5/20.0, 400/1000.0]  # Normalized sample
sample_rgb = [120, 200, 80]  # Sample RGB value
print("Predicted Health Status:", predict_health_status(sample_sensor_data, sample_rgb))


# In[5]:


pip install --upgrade tensorflow


# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data for 3D CNN
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)  # Shape: (samples, 1, 1, features, 1)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    MaxPooling3D(pool_size=(1,1,1)),
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    MaxPooling3D(pool_size=(1,1,1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict health status for new data
def predict_health_status(sensor_data, image_rgb):
    sensor_data = np.array(sensor_data).reshape(1, 1, 1, 4, 1)  # Reshape for model input
    image_rgb = np.array(image_rgb).reshape(1, 1, 1, 3, 1)
    input_data = np.concatenate((image_rgb, sensor_data), axis=3)
    prediction = model.predict(input_data)
    return "Healthy" if prediction < 0.5 else "Diseased"

# Example prediction
sample_sensor_data = [30/40.0, 70/100.0, 5/20.0, 400/1000.0]  # Normalized sample
sample_rgb = [120, 200, 80]  # Sample RGB value
print("Predicted Health Status:", predict_health_status(sample_sensor_data, sample_rgb))


# In[2]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data for 3D CNN
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)  # Shape: (samples, 1, 1, features, 1)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    MaxPooling3D(pool_size=(1,1,1)),
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    MaxPooling3D(pool_size=(1,1,1)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Performance evaluation
predictions = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Additional evaluation metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Diseased"], yticklabels=["Healthy", "Diseased"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict health status for new data
def predict_health_status(sensor_data, image_rgb):
    sensor_data = np.array(sensor_data).reshape(1, 1, 1, 4, 1)  # Reshape for model input
    image_rgb = np.array(image_rgb).reshape(1, 1, 1, 3, 1)
    input_data = np.concatenate((image_rgb, sensor_data), axis=3)
    prediction = model.predict(input_data)
    return "Healthy" if prediction < 0.5 else "Diseased"

# Example prediction
sample_sensor_data = [30/40.0, 70/100.0, 5/20.0, 400/1000.0]  # Normalized sample
sample_rgb = [120, 200, 80]  # Sample RGB value
print("Predicted Health Status:", predict_health_status(sample_sensor_data, sample_rgb))


# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data for 3D CNN
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)  # Shape: (samples, 1, 1, features, 1)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_reduction, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Performance evaluation
predictions = (model.predict(X_test) > 0.5).astype(int)
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Additional evaluation metrics
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Diseased"], yticklabels=["Healthy", "Diseased"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict health status for new data
def predict_health_status(sensor_data, image_rgb):
    sensor_data = np.array(sensor_data).reshape(1, 1, 1, 4, 1)  # Reshape for model input
    image_rgb = np.array(image_rgb).reshape(1, 1, 1, 3, 1)
    input_data = np.concatenate((image_rgb, sensor_data), axis=3)
    prediction = model.predict(input_data)
    return "Healthy" if prediction < 0.5 else "Diseased"

# Example prediction
sample_sensor_data = [30/40.0, 70/100.0, 5/20.0, 400/1000.0]  # Normalized sample
sample_rgb = [120, 200, 80]  # Sample RGB value
print("Predicted Health Status:", predict_health_status(sample_sensor_data, sample_rgb))


# In[4]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
dataset["Temperature (°C)"] = dataset["Temperature (°C)"] / 40.0  # Normalize between 0 and 1
dataset["Humidity (%)"] = dataset["Humidity (%)"] / 100.0
dataset["Rainfall (mm)"] = dataset["Rainfall (mm)"] / 20.0
dataset["CO2 Levels (ppm)"] = dataset["CO2 Levels (ppm)"] / 1000.0

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 0.8) | 
                                     (dataset["Humidity (%)"] > 0.85) | 
                                     (dataset["CO2 Levels (ppm)"] > 0.5) | 
                                     (dataset["Rainfall (mm)"] > 0.5), 1, 0)

# Prepare data
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values)
X_combined = np.concatenate((X_images, X_sensor), axis=1)  # Combine image and sensor data
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Support Vector Machine Classifier
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Evaluation function
def evaluate_model(y_test, predictions, model_name):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    print(f"{model_name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Healthy", "Diseased"], yticklabels=["Healthy", "Diseased"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Evaluate Random Forest
evaluate_model(y_test, rf_predictions, "Random Forest")

# Evaluate SVM
evaluate_model(y_test, svm_predictions, "Support Vector Machine")


# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

# Load dataset
# Assuming a CSV file with columns: 'image_path', 'temperature', 'humidity', 'rainfall', 'co2_levels', 'health_status'
dataset = pd.read_csv('UAV_Tomato_Field_Dataset_Jan2025.csv')

# Parameters
image_size = (224, 224)  # Image size for VGG16
batch_size = 32

# Preprocess images and extract features using VGG16
def extract_image_features(image_paths):
    # Load pre-trained VGG16 model + higher level layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Initialize data generator
    datagen = ImageDataGenerator(rescale=1./255)
    
    features = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = datagen.standardize(img_array)
        feature = model.predict(img_array)
        features.append(feature.flatten())
    
    return np.array(features)

# Extract image features
image_features = extract_image_features(dataset['image_path'])

# Prepare sensor data
sensor_data = dataset[['temperature', 'humidity', 'rainfall', 'co2_levels']].values

# Normalize sensor data
scaler = StandardScaler()
sensor_data = scaler.fit_transform(sensor_data)

# Combine image features and sensor data
X = np.hstack((image_features, sensor_data))
y = dataset['health_status'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Evaluate SVM
svm_predictions = svm_classifier.predict(X_test)
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))
print(f"SVM Accuracy: {accuracy_score(y_test, svm_predictions):.2f}")

# Evaluate Random Forest
rf_predictions = rf_classifier.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_predictions):.2f}")


# In[7]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
scaler = StandardScaler()
dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]] = scaler.fit_transform(
    dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]])

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 1) | 
                                     (dataset["Humidity (%)"] > 1) | 
                                     (dataset["CO2 Levels (ppm)"] > 1) | 
                                     (dataset["Rainfall (mm)"] > 1), 1, 0)

# Prepare data for models
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input for 3D CNN
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_reduction, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"3D CNN Test Accuracy: {accuracy * 100:.2f}%")

# SVM and RF models
X_sensor_train, X_sensor_test, y_train, y_test = train_test_split(X_sensor, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_sensor_train, y_train)
svm_predictions = svm_model.predict(X_sensor_test)

# Train RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_sensor_train, y_train)
rf_predictions = rf_model.predict(X_sensor_test)

# Evaluate models
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")

evaluate_model("SVM", y_test, svm_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Compare models visually
models = ["3D CNN", "SVM", "Random Forest"]
accuracies = [accuracy * 100, accuracy_score(y_test, svm_predictions) * 100, accuracy_score(y_test, rf_predictions) * 100]
plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.ylim([50, 100])
plt.show()


# In[8]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
scaler = StandardScaler()
dataset[["Temperature (\u00b0C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]] = scaler.fit_transform(
    dataset[["Temperature (\u00b0C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]])

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (\u00b0C)"] > 1) | 
                                     (dataset["Humidity (%)"] > 1) | 
                                     (dataset["CO2 Levels (ppm)"] > 1) | 
                                     (dataset["Rainfall (mm)"] > 1), 1, 0)

# Prepare data for models
X_sensor = dataset[["Temperature (\u00b0C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 1, 3, 1)  # Reshape for 3D CNN

# Concatenate sensor and image data as a single input for 3D CNN
X_combined = np.concatenate((X_images, X_sensor.reshape(-1, 1, 1, 4, 1)), axis=3)
y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Build 3D CNN model
model = Sequential([
    Conv3D(32, kernel_size=(1,1,1), activation='relu', input_shape=(1,1,7,1)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Conv3D(64, kernel_size=(1,1,1), activation='relu'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1,1,1)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_reduction, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"3D CNN Test Accuracy: {accuracy * 100:.2f}%")

# SVM and RF models
X_sensor_train, X_sensor_test, y_train, y_test = train_test_split(X_sensor, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_sensor_train, y_train)
svm_predictions = svm_model.predict(X_sensor_test)

# Train RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_sensor_train, y_train)
rf_predictions = rf_model.predict(X_sensor_test)

# Evaluate models
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

evaluate_model("SVM", y_test, svm_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Compare models visually
models = ["3D CNN", "SVM", "Random Forest"]
accuracies = [accuracy * 100, accuracy_score(y_test, svm_predictions) * 100, accuracy_score(y_test, rf_predictions) * 100]
plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.ylim([50, 100])
plt.show()


# In[10]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
scaler = StandardScaler()
dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]] = scaler.fit_transform(
    dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]])

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 1) | 
                                     (dataset["Humidity (%)"] > 1) | 
                                     (dataset["CO2 Levels (ppm)"] > 1) | 
                                     (dataset["Rainfall (mm)"] > 1), 1, 0)

# Prepare data for models
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 3)  # Reshape for 2D CNN

y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

# Build 2D CNN model
model = Sequential([
    Conv2D(32, kernel_size=(1,1), activation='relu', input_shape=(1, 3, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.3),
    
    Conv2D(64, kernel_size=(1,1), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_reduction, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"2D CNN Test Accuracy: {accuracy * 100:.2f}%")

# SVM and RF models
X_sensor_train, X_sensor_test, y_train, y_test = train_test_split(X_sensor, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_sensor_train, y_train)
svm_predictions = svm_model.predict(X_sensor_test)

# Train RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_sensor_train, y_train)
rf_predictions = rf_model.predict(X_sensor_test)

# Evaluate models
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

evaluate_model("SVM", y_test, svm_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Compare models visually
models = ["2D CNN", "SVM", "Random Forest"]
accuracies = [accuracy * 100, accuracy_score(y_test, svm_predictions) * 100, accuracy_score(y_test, rf_predictions) * 100]
plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.ylim([50, 100])
plt.show()


# In[11]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def proposed_algorithm():
    print("Step 1: Load and preprocess dataset")
    print("Step 2: Convert RGB values and normalize sensor data")
    print("Step 3: Assign labels based on conditions")
    print("Step 4: Prepare data for different models")
    print("Step 5: Train and evaluate 2D CNN model")
    print("Step 6: Train and evaluate SVM and Random Forest models")
    print("Step 7: Compare model performances using metrics and visualizations")

# Load dataset
dataset = pd.read_csv("UAV_Tomato_Field_Dataset_Jan2025.csv")

# Convert RGB values to numerical format
def parse_rgb(rgb_string):
    return np.array(eval(rgb_string))  # Convert string tuple to numpy array

dataset["RGB Values"] = dataset["RGB Values"].apply(parse_rgb)

# Normalize sensor data
scaler = StandardScaler()
dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]] = scaler.fit_transform(
    dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]])

# Assign labels based on conditions (Healthy = 0, Diseased = 1)
dataset["Health Status"] = np.where((dataset["Temperature (°C)"] > 1) | 
                                     (dataset["Humidity (%)"] > 1) | 
                                     (dataset["CO2 Levels (ppm)"] > 1) | 
                                     (dataset["Rainfall (mm)"] > 1), 1, 0)

# Prepare data for models
X_sensor = dataset[["Temperature (°C)", "Humidity (%)", "Rainfall (mm)", "CO2 Levels (ppm)"]].values
X_images = np.stack(dataset["RGB Values"].values).reshape(-1, 1, 3)  # Reshape for 2D CNN

y = dataset["Health Status"].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_images, y, test_size=0.2, random_state=42)

# Build 2D CNN model
model = Sequential([
    Conv2D(32, kernel_size=(1,1), activation='relu', input_shape=(1, 3, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.3),
    
    Conv2D(64, kernel_size=(1,1), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.3),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (binary classification: Healthy/Diseased)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_reduction, early_stopping])

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"2D CNN Test Accuracy: {accuracy * 100:.2f}%")

# SVM and RF models
X_sensor_train, X_sensor_test, y_train, y_test = train_test_split(X_sensor, y, test_size=0.2, random_state=42)

# Train SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_sensor_train, y_train)
svm_predictions = svm_model.predict(X_sensor_test)

# Train RF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_sensor_train, y_train)
rf_predictions = rf_model.predict(X_sensor_test)

# Evaluate models
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{name} Model Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}\n")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Diseased'], yticklabels=['Healthy', 'Diseased'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

evaluate_model("SVM", y_test, svm_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

# Compare models visually
models = ["2D CNN", "SVM", "Random Forest"]
accuracies = [accuracy * 100, accuracy_score(y_test, svm_predictions) * 100, accuracy_score(y_test, rf_predictions) * 100]
plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.ylim([50, 100])
plt.show()

proposed_algorithm()


# In[ ]:




