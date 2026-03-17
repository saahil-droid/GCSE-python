#cd /Users/shilpigupta/Python_Projects/exoplanet_Scanner
#/opt/homebrew/bin/python3.11 tests.py


#---------------------make the data ready for ML model training

#use the comand given above to run this file.

import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load the data
with open('flux_data.pkl', 'rb') as f:
    flux_data = pickle.load(f)
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

target_length = 5000
standardized_data = []

for flux in flux_data: #this line will loop through each flux array in the flux_data list
    # Convert to plain list/array to avoid masked array issues
    flux = np.asarray(flux).flatten() #converts the flux array to a plain numpy array and flattens it to ensure that it is a 1D array, which helps to avoid issues that can arise with maskes arrays.

    
    # Remove NaN values, NaN is a special floating-points value that represents "Not a number". what is this for: sometimes the flux data may contain NaN values due to various reasons such as gaps in the data, measurement errors, or issues during the preprocessing steps. Removing NaN values is important because they can cause problems during model training and evaluation, as most machine learning algorithms cannot handle NaN values directly. By filtering out these NaN values, we ensure that the input data is clean and suitable for training a machine learning model, which can lead to better performance and more accurate predictions.
    flux = flux[~np.isnan(flux)]
    
    if len(flux) >= target_length:
        # Truncate - > truncate means to cut of excess data points
        standardized = flux[:target_length]
    else:
        # Pad with zeros
        standardized = np.pad(flux, (0, target_length - len(flux)), constant_values=0)
    
    standardized_data.append(standardized)

# Convert to numpy array
X = np.array(standardized_data)
y = np.array(labels)

print(f"X shape: {X.shape}") 
print(f"y shape: {y.shape}")

#what is a shape: the shape of an array is a tuple that indicates the number of elements along each dimension of the array. EG: if the x shape is (100, 5000), it means there are 100 samples and each sample has 5000 features (data points in the flux array). 
#a tuple is a data structure in python that is similar to a list but cannot be changed.



# Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set: X={X_train.shape}, y={y_train.shape}")
print(f"Test set: X={X_test.shape}, y={y_test.shape}")
print(f"Training labels distribution: {np.bincount(y_train)}")
print(f"Test labels distribution: {np.bincount(y_test)}")
