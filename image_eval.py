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
import os, random, pickle
from PIL import Image
from scipy.fftpack import fft
from random import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
#root_path = '/Users/jili/Box Sync/speechRec/'
root_path='/Users/jingzhuyan/Documents/kaggle'
img_path_test = root_path+'/image/test/'

# Parameters
BATCH_SIZE=128
STEP=10
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
all_files = [el for el in os.listdir(img_path_test) if '.pkl' in el]
shuffle(all_files)

#====generate image batch==== 
def get_batch(BATCH_SIZE):
    while True:            
        j=0
        while (j+1)*BATCH_SIZE < len(all_files):
            spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 81]) 
            tg_batch=np.zeros([BATCH_SIZE, NUM_CLASSES])
            for k, file in enumerate(all_files[j*BATCH_SIZE:(j+1)*BATCH_SIZE]):
                padded_spectrogram=pickle.load(open(img_path_test+file,'rb'))
                spec_batch[k,:,:] = padded_spectrogram
                tg=file.split('_')[0]
                tg_batch[k, tg_id[tg]] = 1                
            j+=1
            spec_batch=spec_batch.astype(np.float32)
            tg_batch=tg_batch.astype(np.float32)
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
from sklearn.metrics import classification_report
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



