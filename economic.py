#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:02:35 2018

@author: patrickdemars
"""

import numpy as np

def calculate_loads(lm, a, b, mins, maxs, num_gen):
    """Calculate loads for all generators as a function of lambda.
    lm: lambda
    a, b: coefficients for quadratic curves of the form cost = a^2p + bp + c
    num_gen: number of generators
    
    Returns a list of individual generator outputs. 
    """
    powers = []
    for i in range(num_gen):
        p = (lm - b[i])/a[i]
        if p < mins[i]:
            p = mins[i]
        elif p > maxs[i]:
            p = maxs[i]
        powers.append(p)
    return powers

def lambda_iteration(load, lambda_low, lambda_high, a, b, mins, maxs, epsilon):
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
    total_output = sum(calculate_loads(lambda_high, a, b, mins, maxs, num_gen))
    while abs(total_output - load) > epsilon:  # abs(total_output - load) > epsilon:  # change epsilon
        lambda_mid = (lambda_high + lambda_low)/2
        total_output = sum(calculate_loads(lambda_mid, a, b, mins, maxs, num_gen))
        if total_output - load > 0:
            lambda_high = lambda_mid
        else:
            lambda_low = lambda_mid
    return calculate_loads(lambda_mid, a, b, mins, maxs, num_gen)

def calculate_costs(outputs, a, b, c):
    """Calculate production costs.
    
    Outputs: list of generating outputs
    a, b, c: lists of coefficients for quadratic cost curves
    num_gen: number of generators
    
    Outputs a list of production costs for each unit. 
    """
    num_gen = len(a)
    cost_list = []
    for i in range(num_gen):
        if outputs[i] == 0:
            cost_list.append(0)
        else:      
            # cost_unit = a[i]*(outputs[i]**2) + b[i]*outputs[i]
            cost_unit = a[i] * (outputs[i] ** 2) + b[i] * outputs[i] + c[i]
            cost_list.append(cost_unit)
    return cost_list
