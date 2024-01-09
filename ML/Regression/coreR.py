# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 02:21:32 2023

@author: kayal
"""
import pandas as pd
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import librosa
import resampy

sounds_path='C:/Users/kayal/ML/sounds/'
sound_df=pd.read_csv('C:/Users/kayal/ML/sound_ml.csv')


def get_audio_duration(file_name):
    audio, sample_rate = librosa.load(file_name)
    duration = len(audio) / sample_rate
    return duration

extracted_durations=[]
for index_num,row in tqdm(sound_df.iterrows()):
    file_name = os.path.join(os.path.abspath(sounds_path),'folder'+str(row["folder"])+'\\',str(row["file"]))
    with open(file_name, "r") as file:   
        duration = get_audio_duration(file_name)
        extracted_durations.append([duration])
        


def format_size(size):
    # Ses dosyasının boyutunu alın
    # 1 KB = 1024 byte, 1 MB = 1024 KB, 1 GB = 1024 MB
    suffixes = ['B', 'KB', 'MB', 'GB']
    index = 0
    while size > 1024 and index < len(suffixes) - 1:
        if(suffixes[index]=='MB'):
            size/=1048576
            index += 1
        elif(suffixes[index]=='KB'):
            size/=1024
            index += 1
            
        # return f"{size:.2f} {suffixes[index]}"
        return size
    

    
extracted_format_size=[]
for index_num,row in tqdm(sound_df.iterrows()):
    file_name = os.path.join(os.path.abspath(sounds_path),'folder'+str(row["folder"])+'\\',str(row["file"]))
    with open(file_name, "r") as file: 
        file_size = os.path.getsize(file_name)
        formatSize = format_size(file_size)
        extracted_format_size.append([formatSize])
        

       
for i in extracted_format_size:
    print(i)
    
for i in extracted_durations:
    print(i)
    
X=extracted_format_size
y=extracted_durations

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val=scaler.transform(X_val)
X_test = scaler.transform(X_test)