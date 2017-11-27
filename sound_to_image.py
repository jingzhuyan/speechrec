#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:45:22 2017

@author: jingzhuyan

Code credit: https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft

#%matplotlib inline

#====set file path====
root_path='/Users/jingzhuyan/Documents/kaggle'
audio_path = root_path+'/train/audio/'
img_path_train = root_path+'/image/train/'
img_path_test = root_path+'/image/test/'
#test_audio_path = '../input/test/audio/'

#====create directories for images====
if not os.path.exists(img_path_train):
    os.makedirs(img_path_train)
if not os.path.exists(img_path_test):
    os.makedirs(img_path_test)
'''
# for all targets
subFolderList = []
for x in os.listdir(audio_path):
    if os.path.isdir(audio_path + '/' + x):
        subFolderList.append(x)
        if not os.path.exists(img_path_train + '/' + x):
            os.makedirs(img_path_train +'/'+ x)
'''            
# only for targets in test          
subFolderList = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'])
for x in subFolderList:
    if os.path.isdir(audio_path + '/' + x):
        if not os.path.exists(img_path_train + '/' + x):
            os.makedirs(img_path_train +'/'+ x)

'''
#===pull samples====
# 30 labels + background noise
sample_audio = []
total = 0
for x in subFolderList:
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
    total += len(all_files)
    # collect the first file from each dir
    sample_audio.append(audio_path  + x + '/'+ all_files[0])
    # show file counts
    print('count: %d : %s' % (len(all_files), x ))
print(total)
'''
#====convert to images====
def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, _, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)
'''
# 9 sample plots
fig = plt.figure(figsize=(10,10))
# for each of the samples
for i, filepath in enumerate(sample_audio[:9]):
    # Make subplots
    plt.subplot(3,3,i+1)
    
    # pull the labels
    label = filepath.split('/')[-2]
    plt.title(label)
    
    # create spectogram
    samplerate, test_sound  = wavfile.read(filepath)
    _, spectrogram = log_specgram(test_sound, samplerate)
    
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.axis('off')
'''
def wav2img(wav_path, targetdir='', figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """

    fig = plt.figure(figsize=figsize)    
    # use soundfile library to read in the wave files
    samplerate, test_sound  = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    
    ## create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    output_file = targetdir +'/'+ output_file
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.imsave('%s.png' % output_file, spectrogram)
    plt.close()

def img_dim(wav_path, targetdir='', figsize=(4,4)):
    samplerate, test_sound  = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    return spectrogram.shape[0]
    
img_dim_dict={}
for i, x in enumerate(subFolderList):
    print(i, ':', x)
    img_dim_dict[x]=[]
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
    for file in all_files:#[:20]: # 20 for each label
        wav2img(audio_path + x + '/' + file, img_path_train + x)
        img_dim_dict[x].append(img_dim(audio_path + x + '/' + file, img_path_train + x))

#>>> [max(img_dim_dict[x]) for x in img_dim_dict.keys()]
#[99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
#>>> [min(img_dim_dict[x]) for x in img_dim_dict.keys()]
#[41, 45, 40, 44, 49, 45, 40, 50, 45, 45]