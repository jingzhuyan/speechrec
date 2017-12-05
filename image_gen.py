#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:45:22 2017

@author: jili, jingzhuyan
"""

import matplotlib.pyplot as plt
from matplotlib.backend_bases import RendererBase
from scipy import signal
from scipy.io import wavfile
#import soundfile as sf
import os, pickle, random
import numpy as np
from PIL import Image
from scipy.fftpack import fft
from random import shuffle
#from scipy import misc
#%matplotlib inline

#====set file path====
root_path='/Users/jili/Box Sync/speechRec/'
#root_path='/Users/jingzhuyan/Documents/kaggle'
img_path_train = root_path+'/image/train/'
img_path_test = root_path+'/image/test/'
img_path_validate = root_path+'/image/validate/'


subFolderList =['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes','silence','unknown']
tg_id = dict((subFolderList[i], i) for i in range(len(subFolderList)))

#determines how many to sample from unknow label, and sample from unknown
avg_class_count = int(np.average([len([y for y in os.listdir(img_path_train) if y.startswith(l+'_')])for l in subFolderList[:-2]]))
all_files = [y for y in os.listdir(img_path_train) if (('.pkl' in y) & (~y.startswith('unknown_')))]
all_files_unk = [y for y in os.listdir(img_path_train) if (('.pkl' in y)& (y.startswith('unknown_')))]
all_files_unk = [ all_files_unk[i] for i in sorted(random.sample(range(len(all_files_unk)), avg_class_count)) ]
all_files += all_files_unk
shuffle(all_files)

#====generate image batch==== 
def get_batch(BATCH_SIZE):
    while True:            
        j=0
        while (j+1)*BATCH_SIZE < len(all_files):
            spec_batch=np.zeros(shape=[BATCH_SIZE, 99, 81]) 
            tg_batch=np.zeros([BATCH_SIZE, NUM_CLASSES])
            for k, file in enumerate(all_files[j*BATCH_SIZE:(j+1)*BATCH_SIZE]):
                #padded_spectrogram = misc.imread(img_path_train+file,mode='I')
                padded_spectrogram=pickle.load(open(img_path_train+file,'rb'))
                spec_batch[k,:,:] = padded_spectrogram
                tg=file.split('_')[0]
                tg_batch[k, tg_id[tg]] = 1                
            j+=1
#            print(spec_batch.shape)
            yield spec_batch, tg_batch

'''
gen=get_batch(128)
s, t = next(gen)
'''

'''
3. MFCC
4. Add background noise ?
'''






            