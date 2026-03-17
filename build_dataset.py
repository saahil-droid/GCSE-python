
# ==============================================================================

import lightkurve as lk
import os
import numpy as np
import astroquery
import pandas as pd
import numpy as np
import pickle

#----------------------------------LOAD AND FILTER THE KOI TABLE--------------------------------------------------------------------

# Load the KOI table
koi_df = pd.read_csv('all_koi_table.csv')
print(f"Total KOIs loaded: {len(koi_df)}")

# Filter for confirmed and false positives only
confirmed = koi_df[koi_df['koi_disposition'] == 'CONFIRMED']
false_pos = koi_df[koi_df['koi_disposition'] == 'FALSE POSITIVE']

print(f"Confirmed: {len(confirmed)}") 
print(f"False Positives: {len(false_pos)}")


#-----------------------------------BUILD ML TRAINING DATASET BY LOOPING THROUGH KIOS-------------------------------------------------

#load the data set
koi_pd = pd.read_csv('all_koi_table.csv')

# Take first 500 of each
confirmed_sample = confirmed ##head(500)
false_pos_sample = false_pos #.head(500)

# Combine them
dataset = pd.concat([confirmed_sample, false_pos_sample]) #the concat function is used to combine the two dataframes

print(f"Dataset size: {len(dataset)}")
print(dataset.head())
#=======================================================================================================================================
#----------------------------------------------------------------------------------test 1: not in this file but in a separate test file, just to test the processing steps on one example before looping through the entire dataset
#======================================================================================================================================
# Now loop through the entire dataset to build training data ---> full 100 rows (50 confirmed + 50 false positives)
flux_data = []
labels = []

for idx, (i, row) in enumerate(dataset.iterrows()):
    kepid = row['kepid']
    period = row['koi_period']
    label = 1 if row['koi_disposition'] == 'CONFIRMED' else 0
    
    try:
        print(f"Processing {idx+1}/7585: KepID {kepid}...")
        lc = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler")[0].download()
        lc = lc.remove_outliers(sigma=5)
        lc_flat = lc.flatten(window_length=101)
        folded = lc_flat.fold(period=period)
        binned = folded.bin(time_bin_size=0.002)
        flux_array = binned.flux.value
        #flux_data.append(flux_array)

        

        # Normalize: subtract mean, divide by std
        flux_normalized = (flux_array - np.mean(flux_array)) / np.std(flux_array)

        flux_data.append(flux_normalized)
        labels.append(label)


        
        # Save every 500 samples
        if (len(flux_data)) % 500 == 0:
            with open(f'checkpoint_{flux_data}.pkl', 'wb') as f:
                pickle.dump({'flux': flux_data, 'labels': labels}, f)
            print(f"Checkpoint saved at {len(flux_data)} samples")

            
            
    except Exception as e:
        print(f"Error with KepID {kepid}: {e}. Skipping...")
        continue

#-----------------------------store data using pickle due to some error.

print(f"\nTotal samples: {len(flux_data)}") #print the total number of samples processed, which should be 100 (50 confirmed + 50 false positives)
print(f"Total labels: {len(labels)}") #print the total number of labels, which should also be 100, confirming that we have a label for each sample
print(f"Example flux array shape: {flux_data[0].shape}") #print the shape of the first flux array to verify that the data has been processed correctly and to understand the dimensionality of the input features for the machine learning model

# Check array lengths
lengths = [len(f) for f in flux_data] #print the lengths of each flux array to check for consistency, as varying lengths could indicate issues with the preprocessing steps or the data itself
print(f"Min length: {min(lengths)}, Max length: {max(lengths)}") #print the minimum and maximum lengths of the flux arrays to identify if there are any significant discrepancies in the data, which could affect model training and may require additional preprocessing steps such as padding or truncation to ensure uniform input sizes for the machine learning model

# Save with pickle
with open('flux_dataFULL.pkl', 'wb') as f: #save the flux_data list to a file named 'flux_data.pkl' using pickle, which allows for saving complex data structures such as lists of arrays without needing to worry about varying lengths or data types, making it a convenient option for storing the processed light curve data for later use in training a machine learning model
    pickle.dump(flux_data, f) #dump the flux_data list into the file using pickle, which serializes the data and allows for easy loading later without needing to worry about the structure or format of the data, making it a good choice for saving the processed light curve data for machine learning purposes
with open('labelsFULL.pkl', 'wb') as f: #save the labels list to a file named 'labels.pkl' using pickle, which allows for easy storage and retrieval of the target variable for training a machine learning model, ensuring that the labels are preserved in their original format and can be easily loaded later for model training and evaluation
    pickle.dump(labels, f) #dump the labels list into the file using pickle, which serializes the data and allows for easy loading later without needing to worry about the structure or format of the data, making it a good choice for saving the target variable for machine learning purposes
    
    # 'wb' stands for "write binary", which is necessary when using pickle to save data, as it ensures that the data is written in a format that can be correctly read back when loading the data later, preserving the integrity of the saved data and allowing for successful deserialization when the data is loaded back into Python for use in training a machine learning model or for other analysis purposes

print("Saved flux_dataFULL.pkl and labelsFULL.pkl")


