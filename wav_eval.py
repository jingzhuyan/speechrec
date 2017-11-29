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
STEP=10

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

'''
batch_gen_val=get_val_batch(vecs, tg, BATCH_SIZE)
ttloss=0
valprob=np.zeros((len(data), NUM_CATES))
for i in range(STEP):
    index_batch, wd_wt_batch, tg_batch = next(batch_gen_val) 
    wd_id_batch=index_batch[:,1]
    feed_dict={
             input_indices: index_batch, 
             input_wd_ids: wd_id_batch,
             input_wd_wts: wd_wt_batch,
             targets: tg_batch}       
    batch_loss=sess.run(loss, feed_dict)
    ttloss += batch_loss
    batch_prob=sess.run(probs, feed_dict)
    valprob[i*BATCH_SIZE:(i+1)*BATCH_SIZE,]=batch_prob  
valloss = ttloss/STEP    
print('Validation loss: %f' %(valloss))

# Evaluation
from sklearn.metrics import roc_auc_score
s=0
cnt=0
for k in range(tg.shape[1]):
    if sum(tg[:,k]) == 0:
        cnt+=1
    else:
        auc=roc_auc_score(tg[:,k], valprob[:,k])
        print(auc)
        s+=auc
print('Ave AUC: %f' %(s/(tg.shape[1]-cnt)))


id_cate=dict([(val,key) for (key,val) in cate_id.items()])
predid=[]
predcate=[]
truecate=[]
for i in range(valprob.shape[0]):
    temp=np.where(valprob[i]>0.5)[0]
    if len(temp)==0:
        temp=np.array([np.argmax(valprob[i])])
    predid.append(temp)
    predcate.append([id_cate[el] for el in temp])
    temp=np.where(tg[i]==1)[0]
    truecate.append([id_cate[el] for el in temp])
df=pd.DataFrame({'true':truecate, 'pred':predcate})
df['tx'] = tx
'''

'''
l=[len(el) for el in tx]
from scipy import stats
stats.describe(l)
'''



