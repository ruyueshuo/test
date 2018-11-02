#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 16:36:13 2018

@author: patrickdemars
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import string
import matplotlib.pyplot as plt
import time

import nn
import rm2
import environment
import helpers
import priority


# -*- coding: UTF-8 -*-


def simulate(epsilon = 1000, alpha = 0.1, temperature = 1, schedule_length = 24, stochastic = True,
             train = False, test = False, dudek = False):
# def simulate(epsilon=1000, alpha = alpha, temperature=temperature, schedule_length=24, stochastic=stochastic,
#              train=False, test=False, dudek=False):
    env.init_state()
    if train:
        for h in range(schedule_length):
            binary_state = helpers.binary_features(env, env.state[0][:num_gen])
            for k in range(5):  # why 5? lookahead =5?
                next_idx = (h + k) % schedule_length
                if stochastic:
                    binary_state = np.append(binary_state, forecast_norm[next_idx])
                else:
                    binary_state = np.append(binary_state, env.demand_norm[next_idx])
            if stochastic:
                binary_state = np.append(binary_state, env.wind_norm[next_idx])
            if h == (schedule_length - 1):
                terminal = True
            else:
                terminal = False
            qs = nn.get_q_values(binary_state.reshape(1, 4 * num_gen + state_size))
            qs_ = env.get_actions(qs)
            action_idx, action = helpers.softmax_policy(env, qs_, temperature)
            # action_idx, action = helpers.epsilon_greedy(env, qs_, epsilon)
            demand = env.state[0][num_gen]
            dispatch = env.get_dispatch(action, demand)
            reward = env.get_reward(dispatch, demand, action)
            rm.add(binary_state, qs, action_idx, reward, terminal)
            if rm.is_full():
                rm.normalise()
                rm.update_all_q_values(alpha)
                nn.optimize()
                rm.reset()
            env.update_state(action, h, schedule_length)
        return np.mean(rm.estimation_errors)
    elif test:
        episode_reward = 0
        test_v = []
        for h in range(schedule_length):
            binary_state = helpers.binary_features(env, env.state[0][:num_gen])
            for k in range(5):
                next_idx = (h + k) % schedule_length
                if stochastic:
                    binary_state = np.append(binary_state, forecast_norm[next_idx])
                else:
                    binary_state = np.append(binary_state, env.demand_norm[next_idx])
            if stochastic:
                binary_state = np.append(binary_state, env.wind_norm[next_idx])
            if h == (schedule_length - 1):
                terminal = True
            else:
                terminal = False
            qs = nn.get_q_values(binary_state.reshape(1, 4 * num_gen + state_size))
            qs_ = env.get_actions(qs)
            action_idx, action = helpers.epsilon_greedy(env, qs_, epsilon)
            demand = env.state[0][num_gen]
            dispatch = env.get_dispatch(action, demand)
            test_v.append(sum(dispatch) - demand)  #add
            episode_reward += env.get_reward(dispatch, demand, action)
            env.update_state(action, h, schedule_length)
            test_states.loc[h] = np.append(env.state[0][:num_gen],
                                           np.append(dispatch, env.get_reward(dispatch, demand, action)))
        return episode_reward, test_v, test_states, forecast_norm


if __name__ == "__main__":

    # Load curve data filename 

    filename = "demand_profiles/net_load_24-2.csv"
    #    filename = "demand_profiles/net_mondays_24.csv"
    #    filename = "demand_profiles/synth_24.csv"

    # Replay memory size & discount factor

    # rm_size = 4800
    rm_size = 4800*3

    # discount_factor = 0.3
    discount_factor = 0.99
    # Setting game parameters

    # num_gen = int(raw_input("how many generators? "))
    # num_gen = int(input("how many generators? "))
    num_gen = 12
    num_actions = (2 ** num_gen)

    # VOLL = 50  # value of lost load
    VOLL = 1000

    # stochastic = True
    stochastic = False

    state_size = 5
    if stochastic:
        state_size += 1

    gen_info = pd.read_csv("unit_details/dudek3.csv", sep=',')
    # gen_info.info()
    # gen_info['e'] = gen_info['e'].astype('float64')
    # gen_info['h'] = gen_info['h'].astype('float64')
    # dt_df = dt_df.convert_objects(convert_numeric=True)
    # dt_df = dt_df.convert_objects(convert_numeric=True)
    # gen_info.info()

    tf.reset_default_graph()

    rm = rm2.ReplayMemory(rm_size,
                          num_actions,
                          num_gen,
                          discount_factor=discount_factor,
                          state_size=state_size)
    nn = nn.NeuralNetwork(num_actions, rm, num_gen, state_size)
    env = environment.Env(num_gen, gen_info, VOLL)
    env.get_times(24)

    all_profiles, all_wind = helpers.get_data(filename)
    # all_profiles = helpers.scale_data(all_profiles, sum(env.max_outputs), 0.2, 0.8)
    # all_profiles, all_wind = all_profiles[500:600], all_wind[500:600]
    #    all_profiles, all_wind = all_profiles[:30], all_wind[:30]
    all_profiles_norm = ((all_profiles - np.min(all_profiles)) /
                         (np.max(all_profiles) - np.min(all_profiles)))
    all_wind_norm = ((all_wind - np.min(all_wind)) /
                     (np.max(all_wind) - np.min(all_wind)))

    rewards = []
    test_rewards = []
    test_actions = []
    test_col_names = list(string.ascii_lowercase[:num_gen * 2 + 1])
    test_states = pd.DataFrame(columns=test_col_names)
    test_voll = []  #add

    estimation_errors = []

    # min_epsilon = 0
    min_epsilon = 0.2
    max_epsilon = 0.85

    min_alpha = 0.01
    max_alpha = 0.3

    min_temperature = 0.05
    max_temperature = 1

    wind_scale = 0.1

    reserve = 0

    # Reserve requirement for PL below

    if stochastic:
        reserve = 0.1  # Tune reserve here

    num_episodes = 10000
    test_episodes = len(all_profiles)

    test_interval = 50

    schedule_length = 24

    # env.demand_raw = np.array(pd.read_csv("demand_profiles/dudek.csv")['demand'], dtype = float)/3
    env.demand_raw = np.array(pd.read_csv("demand_profiles/dudek2.csv")['demand'], dtype=float)
    # env.demand_norm = (env.demand_raw - sum(gen_info.Pmin)) / (sum(gen_info.Pmax) - sum(gen_info.Pmin))
    env.demand_norm = (env.demand_raw - min(env.demand_raw)) / (max(env.demand_raw) - min(env.demand_raw))
    # demand_dudek_norm = (env.demand_raw - min(env.demand_raw)) / (max(env.demand_raw) - min(env.demand_raw))

    start_time = time.time()

    for e in range(num_episodes):
        alpha = helpers.get_alpha(min_alpha, max_alpha, num_episodes, e + 1)
        temperature = round(helpers.get_temperature(min_temperature, max_temperature, num_episodes, e + 1), 3)
        epsilon = helpers.get_epsilon(min_epsilon, max_epsilon, num_episodes, e+1)

        env.get_demand_profile(all_profiles, all_profiles_norm, all_wind_norm, schedule_length)
        if stochastic:
            forecast_norm, forecast_raw = helpers.forecast_demand(env.demand_norm, env.wind_norm,
                                                                  schedule_length, wind_scale,
                                                                  np.min(all_profiles),
                                                                  np.max(all_profiles))
        else:
            forecast_norm, forecast_raw = env.demand_norm, env.demand_raw

        error = simulate(epsilon=epsilon, alpha=alpha, temperature=temperature, stochastic=stochastic, train=True)

        if (e + 1) % test_interval == 0:
            print("episode %s" % (e + 1))
            print("estimation error %s" % error)
            test_reward_total = 0
            pl_reward_total = 0
            for j in range(test_episodes):
                env.demand_raw = all_profiles[j]
                env.demand_norm = all_profiles_norm[j]
                env.wind_norm = all_wind_norm[j]
                if stochastic:
                    forecast_norm, forecast_raw = helpers.forecast_demand(env.demand_norm, env.wind_norm,
                                                                          schedule_length, wind_scale,
                                                                          np.min(all_profiles),
                                                                          np.max(all_profiles))
                else:
                    forecast_norm, forecast_raw = env.demand_norm, env.demand_raw
                ep_reward, test_v, _, _ = simulate(epsilon=1, stochastic=stochastic, test=True)
                test_reward_total += ep_reward
                # test_voll[j][:] = test_v  #add
                pl_reward = priority.run(env.demand_raw, forecast_raw, gen_info, num_gen, reserve)[1]
            print("average cost: %s" % (test_reward_total / test_episodes))
            print('value of lost load: %s' % sum(test_v))
            print(test_v)
            test_rewards.append(test_reward_total / test_episodes)

    end_time = time.time()
    time_taken = end_time - start_time
    print("time taken: %s seconds" % time_taken)

    test_reward_total = 0
    pl_reward_total = 0
    for j in range(test_episodes):

        env.demand_raw = all_profiles[j]
        env.demand_norm = all_profiles_norm[j]
        env.wind_norm = all_wind_norm[j]

        if stochastic:
            forecast_norm, forecast_raw = helpers.forecast_demand(env.demand_norm, env.wind_norm,
                                                                  schedule_length, wind_scale,
                                                                  np.min(all_profiles),
                                                                  np.max(all_profiles))
        else:
            forecast_norm, forecast_raw = env.demand_norm, env.demand_raw

        episode_reward, test_v, _, _ = simulate(epsilon=100, stochastic=stochastic, test=True)
        ep_reward = episode_reward
        test_reward_total += ep_reward
        # test_voll[j] = test_v

        pl_reward = priority.run(env.demand_raw, forecast_raw, gen_info, num_gen, reserve)[1]
        pl_reward_total += pl_reward
    print(test_reward_total / test_episodes)
    print(pl_reward_total / test_episodes)
    print(test_v)
    #
    fig_name = str(time.time()) + ".png"

    if stochastic:
        fig_title = "Imperfect Forecast - Training"
    else:
        fig_title = "Perfect Forecast - Training"

    x = np.arange(1, (num_episodes - 1), test_interval)
    plt.plot(x, test_rewards)
    plt.xlabel("Training Episodes")
    plt.ylabel("Average Reward")
    plt.title(fig_title)
    plt.savefig(fig_name, dpi=1000)

    print(fig_name)
