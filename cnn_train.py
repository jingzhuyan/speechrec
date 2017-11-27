# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:35:10 2017

@author: jili

Tune
1. Loss function sigmoid_cross_entropy_with_logits
2. how many conv and fully connected layers
3. hyperparameters 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import tensorflow as tf
import numpy as np

##############################################
# Define hyperparams                         #
##############################################
LEARNING_RATE=0.1
NUM_CLASSES=10
BATCH_SIZE=128
NUM_EPOCHES=1
NUM_TRAIN_STEPS=1000
EVA_STEP=1
#########################################
# Load data                             #
#########################################
root_path = '/Users/jili/Box Sync/speechRec/'
exec(open(root_path+'speechrec/gen_batch.py').read())

##################################################
# Build a graph                                  #
##################################################
x = tf.placeholder(tf.float32, shape=[None, 99, 161], name='input_layer')
y=tf.placeholder(tf.float32, shape=[BATCH_SIZE,NUM_CLASSES], name='targets')
#y=tf.placeholder(tf.int32, shape=[BATCH_SIZE,NUM_CLASSES], name='targets')

conv = tf.reshape(x, [-1, 99, 161, 1])
for i in range(4):
    conv = tf.layers.conv2d(
                inputs=conv, 
                filters=16*(2**i), 
                kernel_size=[3, 1],
                padding="same",
                activation=tf.nn.relu)
    conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2,2], strides=2)

mpool = tf.reduce_max(conv, axis=[1, 2], keep_dims=True)
apool = tf.reduce_mean(conv, axis=[1, 2], keep_dims=True) # ?
pool = 0.5 * (mpool + apool)
flat = tf.reshape(pool, [-1, 128])

dense_layer = tf.layers.dense(inputs=flat, units=64, activation=tf.nn.relu)
dense_drop = tf.nn.dropout(dense_layer, keep_prob=0.7)

logits = tf.layers.dense(dense_drop, NUM_CLASSES, activation=None, name='logits')
probs=tf.nn.softmax(logits, name='probs')
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits), name='loss')
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits), name='loss') # mean over batch

#optimizer=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
optimizer=tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)

saver = tf.train.Saver()
with tf.Session() as sess:
    print('Graph started...')
    sess.run(tf.global_variables_initializer()) # this could be very slow with large w and large NUM_CATES
    for ep in range(NUM_EPOCHES):
        batch_gen=get_batch(BATCH_SIZE) # define batch generator by train data
        total_loss=0.0
        for step in range(NUM_TRAIN_STEPS):
            batch_x, batch_y = next(batch_gen) # generate a batch 
            loss_batch, _ = sess.run([loss, optimizer], feed_dict={
                                                                 x: batch_x, 
                                                                 y: batch_y})
            total_loss += loss_batch
            # Evaluate Training Data
            if (step+1) % EVA_STEP == 0: # print loss every EVA_STEP
                print('Average loss at Epoch %d and Step %d is: %f' %(ep, step, total_loss/EVA_STEP))
                total_loss=0.0
    saver.save(sess, root_path+'objects/20171120', global_step=0)