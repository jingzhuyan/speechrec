# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:56:04 2017

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os, random, pickle
from PIL import Image
from scipy.fftpack import fft
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
#root_path = '/Users/jili/Box Sync/speechRec/'
root_path='/Users/jingzhuyan/Documents/kaggle'
test_path = root_path+'/test/audio/'

# Parameters
BATCH_SIZE=128
STEP=12
NUM_CLASSES = 12

# Load
os.chdir(root_path+'/objects/20171204/')
sess = tf.Session()
saver=tf.train.import_meta_graph('1-0.meta')
saver.restore(sess, tf.train.latest_checkpoint(root_path+'/objects/20171204/'))

# Restore placeholders
graph=tf.get_default_graph()
x=graph.get_tensor_by_name('input_layer:0')
y=graph.get_tensor_by_name('targets:0')
# Restore variables
loss=graph.get_tensor_by_name('loss:0')
probs=graph.get_tensor_by_name('probs:0')

#==== Prepare to run graph on test set ==== 
subFolderList =['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes','silence','unknown']
tg_id = dict((subFolderList[i], i) for i in range(len(subFolderList)))
all_files = [el for el in os.listdir(test_path) if '.wav' in el]
#all_files=all_files[:1000]
#shuffle(all_files)

#====generate image batch==== 
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
    #while True:            
    j=0
    while (j+1)*BATCH_SIZE < len(all_files):
        spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 81]) 
        #tg_batch=np.zeros([BATCH_SIZE, NUM_CLASSES])
        for k, file in enumerate(all_files[j*BATCH_SIZE:(j+1)*BATCH_SIZE]):
            wav_path = test_path + file
            samplerate, test_sound  = wavfile.read(wav_path)
            new_samplerate = 8000 # resample, the original is 16000
            resampled = signal.resample(test_sound, int(float(new_samplerate)/samplerate * test_sound.shape[0]))
            _, spectrogram = log_specgram(resampled, new_samplerate)
            padded_spectrogram=np.zeros([99,81])
            padded_spectrogram[:min(spectrogram.shape[0],99), :min(spectrogram.shape[1],81)]=spectrogram                
            spec_batch[k,:,:] = padded_spectrogram
        j+=1
        spec_batch=spec_batch.astype(np.float32)
        yield spec_batch
    # handling the last batch
    if (j+1)*BATCH_SIZE >= len(all_files):
        spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 81])  #j was added by 1 at the end of last loop
        for k, file in enumerate(all_files[j*BATCH_SIZE:]):
            wav_path = test_path + file
            samplerate, test_sound  = wavfile.read(wav_path)
            new_samplerate = 8000 # the original is 16000
            resampled = signal.resample(test_sound, int(float(new_samplerate)/samplerate * test_sound.shape[0]))
            _, spectrogram = log_specgram(resampled, new_samplerate)
            padded_spectrogram=np.zeros([99,81])
            padded_spectrogram[:min(spectrogram.shape[0],99), :min(spectrogram.shape[1],81)]=spectrogram                
            spec_batch[k,:,:] = padded_spectrogram
        spec_batch[k+1:,:,:]=np.zeros([99,81]) # fill by 0
        spec_batch=spec_batch.astype(np.float32)
        yield spec_batch
    
print('start prediction')
batch_gen=get_batch(BATCH_SIZE)
valprob=np.zeros(((len(all_files)//BATCH_SIZE+1)*BATCH_SIZE, NUM_CLASSES))
for i in range(len(all_files)//BATCH_SIZE+1):
    if i%10==0:
        print('batch %i' %(i))
    batch_x= next(batch_gen) 
    feed_dict={x: batch_x, y:np.zeros([BATCH_SIZE, NUM_CLASSES])}   
    batch_prob=sess.run(probs, feed_dict)
    valprob[i*BATCH_SIZE:(i+1)*BATCH_SIZE,]=batch_prob  


# Convert id to tg
id_tg=dict([(val,key) for (key,val) in tg_id.items()])
predid=[]
predtg=[]
for i in range(valprob.shape[0]):
    pred=np.argmax(valprob[i])
    predid.append(pred)
    predtg.append(id_tg[pred])
    
result=pd.DataFrame([all_files,predtg[:len(all_files)]]).transpose()
result.columns=['fname','label']
result.to_csv(root_path+'/'+'submission.csv',index=False)