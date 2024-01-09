# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 22:57:19 2023

@author: kayal
"""

import os
import subprocess

input_dir = r'C:\Users\Aslı\Desktop'
output_dir = r'C:\Users\Aslı\Desktop\ML_odev\sounds\Wmachines'
path_to_ffmpeg_exe = r'C:\PATH_Programs\ffmpeg.exe'

files_list = []

for filename in os.listdir(input_dir):
    if filename.endswith('.mp3'):
        files_list.append(filename)

for file_nm in files_list:
    print(file_nm)
    input_file = os.path.join(input_dir, file_nm)
    output_file = os.path.join(output_dir, file_nm.split('.')[0] + '.wav')
    subprocess.call([path_to_ffmpeg_exe, '-i', input_file, output_file])


