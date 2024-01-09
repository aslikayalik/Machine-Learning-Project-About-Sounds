# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 00:32:46 2023

@author: kayal
"""
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
from sklearn.model_selection import train_test_split
import resampy
from sklearn.metrics import adjusted_rand_score


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
extracted_features_df.head()

X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['label'].tolist())

X, y = make_blobs(n_samples=376, centers=5, cluster_std=0.6, random_state=0)

agg_clustering = AgglomerativeClustering(n_clusters=5)  

agg_labels = agg_clustering.fit_predict(X)


# Sonuçları görselleştir
plt.scatter(X[:, 0], X[:, 1], c=agg_labels.astype(float), cmap='viridis')
plt.title('Agglomerative clustering')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Evaulate the clustering performance 
ARI = adjusted_rand_score(y, agg_labels)
print("Adjust Rand Index for Agglomerative:", ARI)






