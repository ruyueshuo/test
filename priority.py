#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:20:44 2018

@author: patrickdemars
"""

import pandas as pd 
import numpy as np 

def get_pl(num_gen, gen_info):  
    """Create a priority list.
    The priority list is created according to the output cost
    in £/MWh when at max output. 
    """ 
    a = gen_info['a'][:num_gen]
    b = gen_info['b'][:num_gen]
    c = gen_info['c'][:num_gen]
    outputs = gen_info['Pmax'][:num_gen]
    cost_max = [] 
    for i in range(num_gen):
        cost_unit = (a[i]*(outputs[i]**2) + b[i]*outputs[i] + c[i])/outputs[i]
        cost_max.append(cost_unit)
    pl = sorted(zip(cost_max, range(num_gen)))
    return pl

def schedule(pl, forecast, state, reserve, gen_info, num_gen):
    """Produce schedule using the PL.
    Solutions are ensured to be feasible in a rolling manner:
    e.g. units committed in hour 1 must be committed for the next
    (Tmin - 1) hours as well. 
    """
    capacity = 0
    headroom = forecast*reserve
    required = forecast + headroom
    uc = np.zeros(num_gen)
    for i in range(num_gen): 
        if -gen_info['Tmin'][i] < state[i] < 0:
            uc[i] = -1
        elif 0 < state[i] < gen_info['Tmin'][i]:
            uc[i] = 1
            capacity += gen_info['Pmax'][i]
    for x in pl:
        unit = x[1]
        if uc[unit] == 0:
            if capacity < required:
                uc[unit] = 1
                capacity += gen_info['Pmax'][unit]
            else:
                break
        else:
            next
    uc = np.where(uc == 1)[0]
    return uc


def calculate_outputs(lm, a, b, mins, maxs, num_gen):
    """Calculate outputs for all generators as a function of lambda.
    lm: lambda
    a, b: coefficients for quadratic curves of the form cost = a^2p + bp + c
    num_gen: number of generators
    
    Returns a list of individual generator outputs. 
    """
    outputs = []
    for i in range(num_gen):
        p = (lm - b[i])/a[i]
        if p < mins[i]:
            p = mins[i]
        elif p > maxs[i]:
            p = maxs[i]
        outputs.append(p)
    return outputs 


def lambda_iteration(load, lambda_low, lambda_high, num_gen, a, b, mins, maxs, epsilon):
    """Calculate economic dispatch using lambda iteration. 
    
    lambda_low, lambda_high: initial lower and upper values for lambda
    a: coefficients for quadratic load curves
    b: constants for quadratic load curves
    epsilon: error as a function 
    
    Returns a list of outputs for the generators.
    """
    num_gen = len(a)
    lambda_low = np.float(lambda_low)
    lambda_high = np.float(lambda_high)
    lambda_mid = 0
    total_output = sum(calculate_outputs(lambda_high, a, b, mins, maxs, num_gen))
    while abs(total_output - load) > epsilon:
        lambda_mid = (lambda_high + lambda_low)/2
        total_output = sum(calculate_outputs(lambda_mid, a, b, mins, maxs, num_gen))
        if total_output - load > 0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid
    return calculate_outputs(lambda_mid, a, b, mins, maxs, num_gen)


def get_dispatch(uc, load, num_gen, gen_info, lambda_low=0, lambda_high=30):
        online_a = np.array(gen_info['a'][uc])
        online_b = np.array(gen_info['b'][uc])
        online_min_outputs = np.array(gen_info['Pmin'][uc])
        online_max_outputs = np.array(gen_info['Pmax'][uc])
        dispatch = np.zeros(num_gen)
        if sum(online_max_outputs) < load:
            econ = online_max_outputs
        elif sum(online_min_outputs) > load:
            econ = online_min_outputs
        else:
            econ = lambda_iteration(
                load, lambda_low, lambda_high, num_gen,
                online_a, online_b, online_min_outputs,
                online_max_outputs, 0.1)
        for i in range(len(uc)):
            dispatch[uc[i]] = econ[i]
        return dispatch
    
    
def calculate_fuel_costs(outputs, num_gen, gen_info):
    """Calculate production costs.
    
    Outputs: list of generating outputs
    a, b, c: lists of coefficients for quadratic cost curves
    num_gen: number of generators
    
    Outputs a list of production costs for each unit. 
    """
    a = gen_info['a'][:num_gen]
    b = gen_info['b'][:num_gen]
    c = gen_info['c'][:num_gen]
    cost_list = []
    for i in range(num_gen):
        if outputs[i] == 0:
            cost_list.append(0)
        else:      
            cost_unit = a[i]*(outputs[i]**2) + b[i]*outputs[i] + c[i]
            cost_list.append(cost_unit)
    return cost_list

def calculate_start_costs(state, action, gen_info):
    start_cost = 0
    for i,a in enumerate(action):
        if a == 1 and state[i] < 0:
            start_i = (np.sum(gen_info['e'][i]*np.exp(gen_info['g'][i]*state[i])) + 
                      np.sum(gen_info['f'][i]*np.exp(gen_info['h'][i]*state[i])))   
            start_cost += start_i
    return start_cost


def update_state(state, action, num_gen):
    """Update the state vector based on action taken."""
    for i in range(num_gen):
        if action[i] == 0:
            if state[i] < 0:
                state[i] += -1
            else:
                state[i] = -1
        else:
            if state[i] > 0:
                state[i] += 1
            else:
                state[i] = 1
    return state


def run(profile, forecast, gen_info, num_gen, reserve):
    all_cost = []
    state = np.ones(num_gen)*-5
    final_schedule = np.zeros((24, num_gen))
    for h,(x,y) in enumerate(zip(forecast, profile)):
        pl = get_pl(num_gen, gen_info)
        uc = schedule(pl, x, state, reserve, gen_info, num_gen)
    
        
        action = np.zeros(num_gen)
        action[uc] = 1
        final_schedule[h] = action
        
        dispatch = get_dispatch(uc, y, num_gen, gen_info)
        fuel_cost = sum(calculate_fuel_costs(dispatch, num_gen, gen_info))
        start_cost = calculate_start_costs(state, action, gen_info)
        cost = fuel_cost + start_cost
        
        update_state(state, action, num_gen)
        
        all_cost.append(cost)
    return final_schedule.T, sum(all_cost)

if __name__ == "__main__":
    filename = "unit_details/dudek3.csv"
    gen_info = pd.read_csv(filename)
    num_gen = 12
    reserve = 0
    a, b = run(profile, forecast, gen_info, num_gen, reserve)
    print(a)
    print(b)

