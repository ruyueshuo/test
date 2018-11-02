#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:09:22 2018

@author: patrickdemars
"""

import numpy as np


class ReplayMemory:
    
    def __init__(self, size, num_actions, num_gen, discount_factor, state_size):

        self.size = size
        self.num_gen = num_gen
        self.states = np.zeros(shape = [size] + [num_gen*4 + state_size], dtype = np.float)
        self.q_values = np.zeros(shape = [size, num_actions], dtype = np.float)
        self.q_values_old = np.zeros(shape = [size, num_actions], dtype = np.float)
        self.actions = np.zeros(shape = size, dtype = np.int)
        self.rewards = np.zeros(shape = size, dtype = np.float)
        self.terminal = np.zeros(shape = size, dtype = bool)
        self.estimation_errors = np.zeros(shape = size, dtype = np.float)
        self.priority = np.zeros(shape = size, dtype = np.float)
        self.discount_factor = discount_factor
        self.num_used = 0
        self.error_threshold = 0.1
        
    def is_full(self):
        return self.num_used == self.size

    def reset(self):
        self.num_used = 0
        
    def add(self, state, q_values, action_idx, reward, terminal):
        if not self.is_full():
            k = self.num_used
            self.num_used += 1
            self.states[k] = state
            self.q_values[k] = q_values
            self.actions[k] = action_idx
            self.rewards[k] = reward
            self.terminal[k] = terminal
    
    def update_all_q_values(self, alpha):
        self.q_values_old[:] = self.q_values[:]
        for k in reversed(range(self.num_used - 1)):
            action = self.actions[k]
            reward = self.rewards[k]
            if self.terminal[k]:
                action_value = reward
            else:
                action_value = reward + self.discount_factor * np.max(self.q_values[k+1]) #Q-learning update
               # action_value = reward + self.discount_factor * (self.q_values[k+1][self.actions[k+1]]) #SARSA update
            self.estimation_errors[k] = abs(action_value - self.q_values[k, action])
            self.q_values[k,action] = (self.q_values[k,action] + 
                         alpha*(action_value - self.q_values[k, action]))
        p = self.estimation_errors**0.5
        self.priority = p/sum(p)
            
    
    def random_batch(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace = False, p = self.priority)
        states_batch = self.states[idx]
        q_values_batch = self.q_values[idx]
        return states_batch, q_values_batch
    
    def normalise(self):
        self.rewards = (self.rewards - np.min(self.rewards))/(np.max(self.rewards) - np.min(self.rewards))
#        self.q_values = (self.q_values - np.amin(self.q_values))/(np.amax(self.q_values) - np.amin(self.q_values))
        
    def standardise(self):
        self.rewards = (self.rewards - np.mean(self.rewards))/np.std(self.rewards)
        self.q_values = (self.q_values - np.mean(self.q_values))/np.std(self.q_values)
