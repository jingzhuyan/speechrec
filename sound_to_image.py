#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:45:22 2017
@author: jingzhuyan
Partial code credit: https://www.kaggle.com/timolee/audio-data-conversion-to-images-eda
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os, pickle
import numpy as np
from PIL import Image
from scipy.fftpack import fft

#%matplotlib inline

#====set file path====
root_path='/Users/jingzhuyan/Documents/kaggle'
#root_path='/Users/jili/Box Sync/speechRec/'
#root_path='/Users/jingzhuyan/Documents/GitHub'
audio_path = root_path+'/train/audio/'
img_path_train = root_path+'/image/train/'
img_path_test = root_path+'/image/test/'
img_path_validate = root_path+'/image/validate/'
#test_audio_path = '../input/test/audio/'

#====read validation & test set====
# no overlap btw validation & test set
text_file = open(root_path+'/train/validation_list.txt', "r")
validation_list = text_file.read().split('\n')
text_file = open(root_path+'/train/testing_list.txt', "r")
testing_list = text_file.read().split('\n')

#====create directories for images====
if not os.path.exists(img_path_train):
    os.makedirs(img_path_train)
if not os.path.exists(img_path_test):
    os.makedirs(img_path_test)
if not os.path.exists(img_path_validate):
    os.makedirs(img_path_validate)
        
# only for targets in test          
#subFolderList = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes','silence'])
subFolderList=sorted(list(set(os.listdir(audio_path)) - set(['.DS_Store','_background_noise_'])))
unknownList=list(set(subFolderList)
                -set(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes','silence']))
#for x in subFolderList:
#    if os.path.isdir(audio_path + '/' + x):
#        #subFolderList.append(x)
#        if not os.path.exists(img_path_train + '/' + x):
#            os.makedirs(img_path_train +'/'+ x)
#if not os.path.exists(img_path_train + '/train_all'):
#    os.makedirs(img_path_train +'/train_all')
#if not os.path.exists(img_path_train + '/silence'):
#    os.makedirs(img_path_train +'/silence')

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


#====cut background noise and save to silence====
def cut_background_noise(wav_path, targetdir='', figsize=(4,4)):
    all_files = [y for y in os.listdir(wav_path) if '.wav' in y]
    i=0
    #fig = plt.figure(figsize=figsize)
    for file in all_files:
        #if file == 'doing_the_dishes.wav':
        #    continue
        print('Cut bkgd noise: %s' %(file))
        samplerate, samples  = wavfile.read(wav_path+'/'+file)
        new_samplerate = 8000 # the original is 16000
        resampled = signal.resample(samples, int(float(new_samplerate)/samplerate * samples.shape[0]))
        _, spectrogram = log_specgram(resampled, new_samplerate)
        
        n=int(spectrogram.shape[0]/99) # a full audio is cut to n parts with each part length 99
        for t in range(n):
            i+=1
            cut_spectrogram=spectrogram[99*t:99*(t+1)]
            output_file = targetdir +'/silence_'+ str(i)
            pickle.dump(cut_spectrogram,open('%s.pkl' % output_file,'wb'))
            #plt.imsave('%s.png' % output_file, cut_spectrogram)
            #plt.close()
        for t in range(n-1):
            i+=1
            cut_spectrogram=spectrogram[99*t+33:99*(t+1)+33]
            output_file = targetdir +'/silence_'+ str(i)
            pickle.dump(cut_spectrogram,open('%s.pkl' % output_file,'wb'))
            #plt.imsave('%s.png' % output_file, cut_spectrogram)
            #plt.close()
        for t in range(n-1):
            i+=1
            cut_spectrogram=spectrogram[99*t+66:99*(t+1)+66]
            output_file = targetdir +'/silence_'+ str(i)
            pickle.dump(cut_spectrogram,open('%s.pkl' % output_file,'wb'))
            #plt.imsave('%s.png' % output_file, cut_spectrogram)
            #plt.close()
        #break # comment to generate all
        
cut_background_noise(audio_path+'_background_noise_',img_path_train)

#====convert to images====
def wav2img(wav_path, targetdir='', figsize=(4,4)):
    """
    takes in wave file path
    and the fig size. Default 4,4 will make images 288 x 288
    """
    #fig = plt.figure(figsize=figsize)    
    # use soundfile library to read in the wave files
    samplerate, samples  = wavfile.read(wav_path)
    new_samplerate = 8000 # the original is 16000
    resampled = signal.resample(samples, int(float(new_samplerate)/samplerate * samples.shape[0]))
    _, spectrogram = log_specgram(resampled, new_samplerate)
    padded_spectrogram=np.zeros([99,81])
    #padded_spectrogram[:min(spectrogram.shape[0],99), :min(spectrogram.shape[1],161)]=0 #[99,161] is the max dims
    padded_spectrogram[:spectrogram.shape[0], :spectrogram.shape[1]]=spectrogram
    ## create output path
    output_file = wav_path.split('/')[-1].split('.wav')[0]
    target_name = wav_path.split('/')[-2]
    # If target_name in unknown list, save file as unknown_filename
    if target_name in unknownList:
        target_name = 'unknown'
    output_file = targetdir +'/'+ target_name + '_'+ output_file
    #scipy.misc.imsave('%s.png' % output_file, padded_spectrogram)
    pickle.dump(padded_spectrogram,open('%s.pkl' % output_file,'wb'))
    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    #plt.imsave('%s.png' % output_file, padded_spectrogram)
    #plt.close()

def img_dim(wav_path, targetdir='', figsize=(4,4)):
    samplerate, test_sound  = wavfile.read(wav_path)
    _, spectrogram = log_specgram(test_sound, samplerate)
    return spectrogram.shape[0]

#====convert to image and save, separate train, validate, test====  
print('started generating pickle files...')
img_dim_dict={}
for i, x in enumerate(subFolderList):
    print(i, ':', x)
    img_dim_dict[x]=[]
    # get all the wave files
    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
    for file in all_files: #100 images for each label
        if (x + '/' + file) in validation_list:
            wav2img(audio_path + x + '/' + file, img_path_validate)
        elif (x + '/' + file) in testing_list:
            wav2img(audio_path + x + '/' + file, img_path_test)
        else:
            wav2img(audio_path + x + '/' + file, img_path_train)#x #save in one folder
        #img_dim_dict[x].append(img_dim(audio_path + x + '/' + file, img_path_train + x))


#>>> [max(img_dim_dict[x]) for x in img_dim_dict.keys()]
#[99, 99, 99, 99, 99, 99, 99, 99, 99, 99]
#>>> [min(img_dim_dict[x]) for x in img_dim_dict.keys()]
#[41, 45, 40, 44, 49, 45, 40, 50, 45, 45]