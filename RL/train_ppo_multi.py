"""
train.py
solve the tdsp with single graph
separate the training by curriculum (number of hops).
    - for different graph scale, can have different number of curriculum
"""

import argparse
import copy
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from scipy.signal import savgol_filter
import os

import dgl
import torch
from torch.optim import Adam

# from utils import cal_actual_travel_time
from utils_log import createLogger
from utils_env import GNN_ENV_MULTI, MLP_ENV_MULTI, cal_reward_to_go, \
    remove_temp_model, test_gnn_env_travel_time

from utils import load_baseline_result

from agent import PPO_GNN, PPO_MLP, PPO_GNN_new
from modelzoo import PPOActorCritic_GNN, PPOActorCritic_MLP, PPOActorCritic_GNN_new

torch.set_default_dtype(torch.float64)

log = createLogger(__name__)
log.info('in train_ppo_multi.py')

parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default='None')
parser.add_argument("--map_name", type=str, nargs='+', help="folder name")
parser.add_argument('--graph_size_list', type=int, nargs='+')

parser.add_argument('--num_stop_action_list', type=int, nargs='+')
parser.add_argument('--time_interval_list', type=float, nargs='+')

parser.add_argument('--num_episodes_list', type=int, nargs='+')
parser.add_argument('--num_test', default=10, type=int)
parser.add_argument('--print_episode', default=10, type=int)
parser.add_argument('--test_episode', default=100, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gamma', default=0.9, type=float)

parser.add_argument('--hidden_channels', default=64, type=int)

parser.add_argument('--agent_type', type=str)

args = parser.parse_args()
num_episodes_list = args.num_episodes_list
num_stop_action_list = args.num_stop_action_list

length = len(args.map_name)
assert(length == len(args.graph_size_list))
assert(length == len(args.num_stop_action_list))
assert(length == len(args.time_interval_list))
assert(length == len(args.num_episodes_list))

# dgl.seed(234)
# torch.manual_seed(234)
# np.random.seed(234)

if args.agent_type == 'GNN':
    ENV = GNN_ENV_MULTI# PPO_GNN_ENV #PPO_MLP_ENV GNN_ENV_MULTI
    AGENT = PPO_GNN# PPO_GNN #PPO_MLP
    MODEL = PPOActorCritic_GNN# PPOActorCritic_GNN #PPOActorCritic_MLP

if args.agent_type == 'GNN_new':
    ENV = GNN_ENV_MULTI# PPO_GNN_ENV #PPO_MLP_ENV GNN_ENV_MULTI
    AGENT = PPO_GNN_new# PPO_GNN #PPO_MLP
    MODEL = PPOActorCritic_GNN_new# PPOActorCritic_GNN #PPOActorCritic_MLP
    
if args.agent_type == 'MLP':
    ENV = MLP_ENV_MULTI# PPO_GNN_ENV #PPO_MLP_ENV GNN_ENV_MULTI
    AGENT = PPO_MLP # DQN_MLP #PPO_MLP
    MODEL = PPOActorCritic_MLP# PPOActorCritic_GNN #PPOActorCritic_MLP

# Instantiate the GNN and the agent
model = MODEL(hidden_channels=args.hidden_channels) #GNN(hidden_channels=128) #MLP(hidden_channels=128)
if args.load_model != "None":
    checkpoint = torch.load('model_multi/{}.pt'.format(args.load_model))
    model.load_state_dict(checkpoint['model_state_dict'])
agent = AGENT(args, model) # DDQN_MLP DQN_MLP DQN_GNN

test_env_idx = len(args.map_name)-1
test_env = ENV(args, test_env_idx, test_env=True)

remove_temp_model()
tdsp_result, static_result = load_baseline_result(args)

# Training loop
total_train_reward_list, total_train_loss_list, total_train_action_list = [], [], []
total_test_time_list = []
episode = 0

for env_idx in range(len(args.map_name)):
    try:
        checkpoint = torch.load('model_multi/best_{}_{}_{}.pt'.format(agent.__class__.__name__, args.map_name[env_idx], args.lr))
        agent.model.load_state_dict(checkpoint['model_state_dict'])
    except:
        try:
            checkpoint = torch.load('model_multi/temp_{}_{}_{}.pt'.format(agent.__class__.__name__, args.map_name[env_idx], args.lr))
            agent.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            pass
    
    maximum_improvement = -1e9
    
    # training stage
    # Instantiate the environment
    env = ENV(args, env_idx)
    for _ in range(num_episodes_list[env_idx]):
        episode += 1
        curr_node_list = []
        # print("-------------------------")
        # TODO:
        # (done) change the random walk steps
        # (done) change the MLP features
        state, curr_node = env.reset(train_env=True) # state is the subgraph of the dgl graph.
        curr_node_list.append(curr_node)
        
        # record history in training
        total_train_reward, total_train_travel_time, total_train_loss_episode = 0, 0, []
        ep_states, ep_actions, ep_log_probs, ep_rewards = [], [], [], []
        
        for action_idx in range(num_stop_action_list[env_idx]):  # Limit each episode to a maximum of 100 steps
            # input:    state: subgraph
            # output:   actor feat, critic feat, action, log_prob, reward, new_state

            action, old_log_prob, policy_mask, value_mask = agent.select_action(state, curr_node)
            (next_state, curr_node), (reward, travel_time), done = env.step(action)

            curr_node_list.append(curr_node)        
            ep_states.append(state)
            ep_actions.append(action)
            ep_rewards.append(reward)
            ep_log_probs.append(old_log_prob)
            total_train_reward += reward
            total_train_travel_time += travel_time
            if done:    break
            state = next_state

        # Compute advantages and target ep_values
        ep_reward_rtg = cal_reward_to_go(ep_rewards)
        
        # remove the case only take one action to desitnation
        if len(ep_states) == 1:
            loss = 0.0
            pass
        else:
            loss, actor_loss, critic_loss, entropy_loss = agent.optimize(ep_states, \
                ep_actions, ep_log_probs, ep_reward_rtg, curr_node_list[:-1])
        
        total_train_loss_episode.append(loss)
        total_train_reward_list.append(total_train_reward)
        total_train_loss_list.append(np.mean(total_train_loss_episode))
        total_train_action_list.append(len(curr_node_list)-1)
        
        
        # testing on fixed OD
        if episode % args.print_episode == 0:
            log.info("Training Episode {}: Total reward: {:.2f}, done: {}, actor_loss (-): {:.2f}, critic_loss (-): {:.2f}, step: {}".format(episode, \
                total_train_reward, done, actor_loss, critic_loss, len(curr_node_list)-1), l=1)
        
        if episode % args.test_episode == 0:
            travel_time_per_test_list, done_per_test_list, diff_tdsp, diff_static, improvement = \
                test_gnn_env_travel_time(num_stop_action_list, agent, test_env_idx, test_env, tdsp_result, static_result)
            total_test_time_list.append(travel_time_per_test_list)
            log.info("Testing Episode {}: tdsp_diff (+): {:.2f}, {:.2f}, static_diff (+): {:.2f}, {:.2f}, success: {:.2f}".format(episode, \
                diff_tdsp.mean(), diff_tdsp.std(), diff_static.mean(), diff_static.std(), np.sum(done_per_test_list)/len(done_per_test_list)), l=1)
            log.info("Testing Episode {}: tdsp_diff (+): {:.2f}, static_diff (+): {:.2f}, success: {:.2f}".format(episode, \
                diff_tdsp[done_per_test_list].mean(), diff_static[done_per_test_list].mean(), np.sum(done_per_test_list)/len(done_per_test_list)), l=1)
            
            if np.mean(improvement) > maximum_improvement:
                torch.save({'model_state_dict': agent.model.state_dict()}, \
                    'model_multi/best_{}_{}_{}_{}.pt'.format(agent.__class__.__name__, \
                    args.map_name[env_idx], args.graph_size_list[env_idx], args.lr))
                maximum_improvement = np.mean(improvement)
                log.info("Save the best model, improvement: {}".format(np.mean(improvement)), l=1)
            
    torch.save({'model_state_dict': agent.model.state_dict()}, 'model_multi/{}_{}_{}_{}.pt'.format(agent.__class__.__name__, args.map_name[env_idx], args.graph_size_list[env_idx], args.lr))
    torch.save({'model_state_dict': agent.model.state_dict()}, 'model_multi/temp_{}_{}_{}.pt'.format(agent.__class__.__name__, args.map_name[env_idx], args.lr))