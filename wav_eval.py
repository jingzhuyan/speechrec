# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:35:23 2017

@author: jili
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
import os
from PIL import Image
from scipy.fftpack import fft
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
root_path = '/Users/jili/Box Sync/speechRec/'
audio_path = root_path+'/wav/test_set/'

# Parameters
BATCH_SIZE=128
STEP=20
NUM_CLASSES=10
# Load
os.chdir(root_path+'/objects/20171128/')
sess = tf.Session()
saver=tf.train.import_meta_graph('1-0.meta')
saver.restore(sess, tf.train.latest_checkpoint(root_path+'objects/20171128/'))

# Restore placeholders
graph=tf.get_default_graph()
x=graph.get_tensor_by_name('input_layer:0')
y=graph.get_tensor_by_name('targets:0')
# Restore variables
loss=graph.get_tensor_by_name('loss:0')
probs=graph.get_tensor_by_name('probs:0')

#==== Prepare to run graph on test set ==== 
all_files = [y for y in os.listdir(audio_path) if '.wav' in y]
shuffle(all_files)
subFolderList = sorted(['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'])
tg_id = dict((subFolderList[i], i) for i in range(len(subFolderList)))

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
        j=0
        while (j+1)*BATCH_SIZE < len(all_files):
            spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 161]) 
            tg_batch=np.zeros([BATCH_SIZE, NUM_CLASSES])
            for k, file in enumerate(all_files[j*BATCH_SIZE:(j+1)*BATCH_SIZE]):
                wav_path = audio_path + file
                samplerate, test_sound  = wavfile.read(wav_path)
                _, spectrogram = log_specgram(test_sound, samplerate)
                padded_spectrogram=np.zeros([99,161])
                padded_spectrogram[:min(spectrogram.shape[0],99), :min(spectrogram.shape[1],161)]=spectrogram 
                spec_batch[k,:,:] = padded_spectrogram
                tg=file.split('_')[0]
                tg_batch[k, tg_id[tg]] = 1                
            j+=1
            yield spec_batch, tg_batch


batch_gen=get_batch(BATCH_SIZE)
ttloss=0
valprob=np.zeros((STEP*BATCH_SIZE, NUM_CLASSES))
valtg=np.zeros((STEP*BATCH_SIZE, NUM_CLASSES))
for i in range(STEP):
    batch_x, batch_y = next(batch_gen) 
    feed_dict={x: batch_x, y: batch_y}   
    batch_loss=sess.run(loss, feed_dict)
    ttloss += batch_loss
    batch_prob=sess.run(probs, feed_dict)
    valprob[i*BATCH_SIZE:(i+1)*BATCH_SIZE,]=batch_prob  
    valtg[i*BATCH_SIZE:(i+1)*BATCH_SIZE,]=batch_y
valloss = ttloss/STEP    
print('Validation loss: %f' %(valloss))

## Evaluation
from sklearn.metrics import roc_auc_score, classification_report
tt=0
for k in range(valtg.shape[1]):
    auc=roc_auc_score(valtg[:,k], valprob[:,k])
    print(auc)
    tt+=auc
print('Ave AUC: %f' %(tt/valtg.shape[1]))

# Convert id to tg
id_tg=dict([(val,key) for (key,val) in tg_id.items()])
predid=[]
predtg=[]
truetg=[]
for i in range(valprob.shape[0]):
    pred=np.argmax(valprob[i])
    predid.append(pred)
    predtg.append(id_tg[pred])
    trueid=np.where(valtg[i]==1)[0][0]
    truetg.append(id_tg[trueid])
print(classification_report(truetg, predtg))

'''
# Plot roc curve
from sklearn.metrics import roc_curve 
from matplotlib import pyplot as plt
fpr, tpr, threshold = roc_curve(valtg[:,0],valprob[:,0])
plt.plot(fpr, tpr)
'''

