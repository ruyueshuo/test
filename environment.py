#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 10:53:53 2018

@author: patrickdemars
"""

import itertools
import numpy as np
import pandas as pd
import economic
from helpers import get_day


class Env:
    def __init__(self, num_gen, gen_info, VOLL):
        self.action_space = np.asarray(list(itertools.product([0, 1], repeat=num_gen)))
        self.action_size = 2**num_gen
        self.num_gen = num_gen

        self.cost_a = np.array(gen_info['a'])[:num_gen]
        self.cost_b = np.array(gen_info['b'])[:num_gen]
        self.cost_c = np.array(gen_info['c'])[:num_gen]
        self.cost_e = np.array(gen_info['e'])[:num_gen]
        self.cost_f = np.array(gen_info['f'])[:num_gen]
        self.cost_g = np.array(gen_info['g'])[:num_gen]
        self.cost_h = np.array(gen_info['h'])[:num_gen]
        self.min_outputs = np.array(gen_info['Pmin'])[:num_gen]
        self.max_outputs = np.array(gen_info['Pmax'])[:num_gen]
        self.min_op = np.array(gen_info['Tmin'])[:num_gen]

        self.state = []
        self.times_df = []
        self.demand_raw = []
        self.demand_norm = []
        self.VOLL = VOLL

    def get_times(self, periods):  # 什么意思。。get_times(24)
        time = np.array(range(periods), dtype=float)  # 1,2,3...,23
        sin_time = np.sin(2*np.pi*time/periods)
        cos_time = np.cos(2*np.pi*time/periods)
        time_df = pd.DataFrame(data=dict(sin_time=sin_time, cos_time=cos_time))
        self.times_df = time_df

    def get_demand_raw(self, amin, amax):
        demand_raw = pd.read_csv("profile_agg_2013_2016.csv").iloc[:, 1]
        self.demand_raw = (amax-amin)*demand_raw + amin

    def get_demand_profile(self, profiles_raw, profiles_norm, wind_norm, schedule_length):
        self.demand_raw, self.demand_norm, self.wind_norm = get_day(profiles_raw, profiles_norm, wind_norm)
        self.demand_raw, self.demand_norm, self.wind_norm = (self.demand_raw[:schedule_length], 
                                                             self.demand_norm[:schedule_length], 
                                                             self.wind_norm[:schedule_length])

    def get_dispatch(self, action, demand, lambda_low=0, lambda_high=50):
        idx = np.where(action == 1)[0]
        online_a = self.cost_a[idx]
        online_b = self.cost_b[idx]
        online_min_outputs = self.min_outputs[idx]
        online_max_outputs = self.max_outputs[idx]
        dispatch = np.zeros(self.num_gen)
        if sum(online_max_outputs) < demand:
            econ = online_max_outputs
        elif sum(online_min_outputs) > demand:
            econ = online_min_outputs
        else:
            econ = economic.lambda_iteration(
                demand, lambda_low, lambda_high,
                online_a, online_b, online_min_outputs,
                online_max_outputs, 0.1)
        for i in range(len(idx)):
            dispatch[idx[i]] = econ[i]
        return dispatch

    def get_reward(self, dispatch, demand, action):
        ENS = 0
        cost_fuel = sum(economic.calculate_costs(dispatch, self.cost_a, 
                                                 self.cost_b, self.cost_c))
        on_idx = np.where((action == 1) & (self.state[0][:self.num_gen] < 0))[0]
        cost_start = (np.sum(self.cost_e[on_idx]*np.exp(self.cost_g[on_idx]*self.state[0][on_idx])) + 
                      np.sum(self.cost_f[on_idx]*np.exp(self.cost_h[on_idx]*self.state[0][on_idx])))
        if sum(dispatch) < demand:
            ENS = demand - sum(dispatch)
        elif sum(dispatch) > demand:
            ENS = sum(dispatch) - demand
        reward = -(cost_fuel + cost_start + ENS*self.VOLL)
        return reward

    def init_state(self):
        """Initialise the state.
        Demand is initially set to the 1st value.
        In other words, we have the generation configuration as of 23:00
        and we are looking to see what we want to turn on at 00:00 to satisfy
        the demand at that time.
        At the moment this is set to all off for 6 hours each, outside
        of minimum up time constraint. Could be different.
        """
        gen_status = np.ones(self.num_gen)*-6
        self.state = np.append(gen_status, (self.demand_raw[0],
                               self.times_df.iat[0, 0],
                               self.times_df.iat[0, 1])).reshape(1, self.num_gen + 3)

    def update_state(self, action, h, schedule_length):
        """Update state in hour h.
        Moves state forward according to action.
        """
        if h == schedule_length - 1:
            self.state[0][self.num_gen] = self.demand_raw[0]
            self.state[0][self.num_gen+1] = self.times_df.iat[0, 0]
            self.state[0][self.num_gen+2] = self.times_df.iat[0, 1]
        else:
            self.state[0][self.num_gen] = self.demand_raw[h+1]
            self.state[0][self.num_gen+1] = self.times_df.iat[h+1, 0] 
            self.state[0][self.num_gen+2] = self.times_df.iat[h+1, 1]
        for row in range(len(action)):
            if action[row] == 0:
                if self.state[0][row] <= -1:
                    self.state[0][row] = self.state[0][row] - 1
                else:
                    self.state[0][row] = -1
            else:
                if self.state[0][row] <= -1:
                    self.state[0][row] = 1
                else:
                    self.state[0][row] = self.state[0][row] + 1

    def get_actions(self, q_values):
        qs_copy = np.copy(q_values)
        for k in range(self.num_gen):
            if -(self.min_op[k]-1) <= self.state[0][k] < 0:
                idx = np.where(self.action_space[:, k] == 1)
                qs_copy[0][idx] = np.NaN
            elif 0 < self.state[0][k] <= self.min_op[k] - 1:
                idx = np.where(self.action_space[:, k] == 0)
                qs_copy[0][idx] = np.NaN
        return qs_copy
