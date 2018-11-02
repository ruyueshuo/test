#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:37:27 2018

@author: patrickdemars
"""

import pandas as pd 
import numpy as np

def get_data(filename):
    """Pull demand time series data from .csv file.
    This is set up for days with 24 periods.
    Returns a n x 24 arrays for a .csv with n days.
    The .csv file should be formatted as |period|date|value. 
    """
    df = pd.read_csv(filename)
    net_load = np.array(df['net_load'])
    wind = np.array(df['wind'])
    ndays = int(len(net_load)/24)
    # ndays = ndays.asptye(int)
    print(ndays)
    return net_load.reshape(ndays, 24), wind.reshape(ndays, 24)

def get_day(data_raw, data_norm, wind):
    """Pull a day time series from data.
    Samples at random with replacement. 
    data should be a n x 24 array as returned by get_data.
    """
    idx = np.random.choice(range(len(data_raw)))
    return data_raw[idx], data_norm[idx], wind[idx]

def forecast_demand(actual_demand, wind, length, sd, a, b):
    """Produce a normalised forecast of net load. 
    a, b are the min and max (respectively) of the profiles data. They
    are used for scaling back up from the normalised forecast to raw
    forecast.
    """
    error = np.random.normal(loc = 0.0, scale = wind*sd, size = length)
    forecast_norm = actual_demand + error
    forecast_raw = forecast_norm*(b-a) + a
    return forecast_norm, forecast_raw

def get_epsilon(min_epsilon, max_epsilon, num_episodes, episode):
    """Get epsilon value for given episode.
    Epsilon increases linearly from min_epsilon to max_epsilon
    """
    epsilon = min_epsilon + ((float(episode)/float(num_episodes))*(max_epsilon - min_epsilon))
    return epsilon

def get_alpha(min_alpha, max_alpha, num_episodes, episode):
    """Get epsilon value for given episode.
    Epsilon decreases linearly from min_epsilon to max_epsilon
    """
    alpha = max_alpha - ((float(episode)/float(num_episodes))*(max_alpha - min_alpha))
    return alpha

def get_temperature(min_temp, max_temp, num_episodes, episode):
    """Get epsilon value for given episode.
    Epsilon decreases linearly from min_temp to max_temp
    """
    temp = max_temp - ((float(episode)/float(num_episodes))*(max_temp - min_temp))
    return temp

def scale_data(profile_data, sum_gens, amin, amax):
    """Scale time series data. 
    Scaling is made according to sum_gens; the sum of all generator outputs.
    amin, amax scale the desired range of values. 
    """
    a = sum_gens*amin
    b = sum_gens*amax
    scaled_data = ((b-a)*(profile_data - np.min(profile_data)))/(
            np.max(profile_data) - np.min(profile_data)) + a
    return scaled_data
        
def epsilon_greedy(env, q_values, epsilon):
    """Epsilon greedy policy for choosing actions.
    Returns the index for the chosen action and the action itself.
    The policy chooses a random action (from available actions) with
    probability (1 - epsilon).
    """
    if np.random.rand() < epsilon:
        idx = np.nanargmax(q_values)
    else:
        idx_ = np.where(np.isnan(q_values[0]) == False)
        idx = np.random.choice(idx_[0])
#        idx = np.int(np.where(q_values == np.random.choice(q_values[0][~np.isnan(q_values[0])]))[1])
    return idx, env.action_space[idx]

def softmax_policy(env, q_values, temperature):
    """Softmax policy for choosing actions.
    Returns the index for the chosen action and the action itself. 
    The exploration is determined by the temperature: as temperature -> 0, 
    the policy becomes an argmax over q_values. 
    """
    q_values = np.nan_to_num(q_values)
    softmax = np.exp(q_values/temperature) / np.sum(np.exp(q_values/temperature))
    idx = np.random.choice(range(env.action_size), p = softmax[0])
    return idx, env.action_space[idx]

def binary_features(env, gen_states):   #  0-0, 0-1, 1-0, 1-1？
    features = np.zeros(len(gen_states)*4)
    for i in range(len(gen_states)):
        if gen_states[i] < -env.min_op[i]:
            features[4*i] = 1
        elif -env.min_op[i] <= gen_states[i] < 0:
            features[4*i + 1] = 1
        elif 0 < gen_states[i] <= env.min_op[i]:
            features[4*i + 2] = 1
        else:
            features[4*i + 3] = 1
    return features

def normal_features(env, gen_states):
    features = np.zeros(len(gen_states)*2)
    for i in range(len(gen_states)):
        if gen_states



