#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:10:36 2018

@author: patrickdemars
"""

import tensorflow as tf
import numpy as np

class NeuralNetwork:
    def __init__(self, num_actions, replay_memory, num_gen, state_size):
        self.replay_memory = replay_memory
        self.x = tf.placeholder(dtype = tf.float32, shape = (None, num_gen*4 + state_size))
        self.learning_rate = tf.placeholder(dtype = tf.float32, shape = [])
        self.q_values_new = tf.placeholder(tf.float32, 
                                           shape = [None, num_actions],
                                           name = 'q_values_new')
        
        init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)
        activation = tf.nn.relu
        num_output = num_actions
        
        net = self.x
        net1 = tf.layers.dense(inputs=net, name='layer_fc1', units=64,
                                          kernel_initializer=init, activation=activation, 
                                          reuse = tf.AUTO_REUSE)
        
        net2 = tf.layers.dense(inputs=net1, name='layer_fc2', units=32,
                                          kernel_initializer=init, activation=activation, 
                                          reuse = tf.AUTO_REUSE)
        
        net3 = tf.layers.dense(inputs=net2, name='layer_fc3', units=16,
                                          kernel_initializer=init, activation=activation, 
                                          reuse = tf.AUTO_REUSE)
                
        net4 = tf.layers.dense(inputs=net3, name='layer_fc_out', units=num_output,
                                          kernel_initializer=init, activation=None, 
                                          reuse = tf.AUTO_REUSE)

        self.q_values = net4
        squared_error = tf.square(self.q_values - self.q_values_new)
        sum_squared_error = tf.reduce_sum(squared_error, axis=1)
        self.loss = tf.reduce_mean(sum_squared_error)
        
#        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = tf.nn.softmax(self.q_values_new), logits = self.q_values)
#        self.loss = cross_entropy
#        self.mean_loss = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)            
        self.session = tf.Session()        
        self.session.run(tf.global_variables_initializer())
        
    def get_q_values(self, states):
        feed_dict = {self.x: states}
        values = self.session.run(self.q_values, feed_dict = feed_dict)
        return values

    def optimize(self, min_epochs = 1.0, max_epochs = 10, batch_size = 16,
                 loss_limit = 1e-3, learning_rate = 1e-3):
        
        loss_history = np.zeros(100, dtype=float)
        
        iterations_per_epoch = self.replay_memory.num_used / batch_size
        min_iterations = int(iterations_per_epoch * min_epochs)
        max_iterations = int(iterations_per_epoch * max_epochs)
                
        for i in range(max_iterations):
            
            state_batch, q_values_batch = self.replay_memory.random_batch(batch_size)
            
            feed_dict = {self.x: state_batch,
                         self.q_values_new: q_values_batch,
                         self.learning_rate: learning_rate}
            
            loss_val, _ = self.session.run([self.loss, self.optimizer], 
                                           feed_dict = feed_dict)

            if i%100 == 0:
                print(i, 'loss_ is :', loss_val)

            loss_history = np.roll(loss_history, 1)
            loss_history[0] = loss_val
            loss_mean = np.mean(loss_history)
                                    
            if i > min_iterations and loss_mean < loss_limit:
                break
