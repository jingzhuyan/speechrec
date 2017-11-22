#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:45:22 2017

@author: jingzhuyan
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
root_path='/Users/jili/Box Sync/speechRec/'
#root_path='/Users/jingzhuyan/Documents/GitHub'
audio_path = root_path+'/train/audio/'
#test_audio_path = '../input/test/audio/'

subFolderList = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'])
tg_id = dict((subFolderList[i], i) for i in range(len(subFolderList)))

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

def get_batch(BATCH_SIZE):
    while True:
        for i, x in enumerate(subFolderList):
            print(i, ':', x)
            all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]
            j=0
            while (j+1)*BATCH_SIZE < len(all_files):
                spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 161])
                tg_batch=np.zeros([BATCH_SIZE, NUM_CLASSES])
                for k, file in enumerate(all_files[j*BATCH_SIZE:(j+1)*BATCH_SIZE]):
                    wav_path = audio_path + x + '/' + file
                    samplerate, test_sound  = wavfile.read(wav_path)
                    _, spectrogram = log_specgram(test_sound, samplerate)
                    padded_spectrogram=np.zeros([99,161])
                    padded_spectrogram[:min(spectrogram.shape[0],99), :min(spectrogram.shape[1],161)]=0 #check if [99,161] is the max dims
                    spec_batch[k,:,:] = padded_spectrogram
                j+=1
                tg_batch[:,tg_id[x]]=1
                yield spec_batch, tg_batch

'''
gen=get_batch(128)
s, t = next(gen)
'''




            







            