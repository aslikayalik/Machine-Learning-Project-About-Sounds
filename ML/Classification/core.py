# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 23:06:20 2023

@author: kayal
"""

import pandas as pd
import librosa
from tqdm import tqdm
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import resampy



sounds_path='C:/Users/kayal/ML/sounds/'
sound_df=pd.read_csv('C:/Users/kayal/ML/sound_ml.csv')

def features_extractor(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    return mfccs_scaled_features



extracted_features=[]
expected_size=4
for index_num,row in tqdm(sound_df.iterrows()):
    file_name = os.path.join(os.path.abspath(sounds_path),'folder'+str(row["folder"])+'\\',str(row["file"]))
    file_size = os.path.getsize(file_name)
    if file_size == 0:
        print("Dosya boş!")
        print(file_name)
    elif file_size < expected_size:
        print("Dosya beklenenden daha kısa!")
    else:
        with open(file_name, "r") as file:   
            final_class_labels=row["label"]
            data=features_extractor(file_name)
            extracted_features.append([data,final_class_labels])


extracted_features_df = pd.DataFrame(extracted_features,columns=['feature','label'])
print(extracted_features_df.head())

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['label'].tolist())


# Split train, test and validation set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test = scaler.transform(X_test)

    