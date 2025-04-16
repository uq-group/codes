import dgl
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR
import random

from utils_log import createLogger

log = createLogger(__name__)
log.info('in agent.py')

class PPO_GNN:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        try:
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            # self.scheduler = MultiStepLR(self.optimizer, milestones=[3000, 6000], gamma=0.5)
            self.scheduler = StepLR(self.optimizer, step_size = 4000, gamma=0.1)
            self.gamma = self.args.gamma # Discount factor
        except:
            pass
        
        self.loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.clip_epsilon = 0.2
       
    def select_action(self, state, curr_node):
        policy_mask = self.model.actor_forward(state, curr_node=curr_node)
        value_mask = self.model.critic_forward(state)
        # policy_mask, value_mask = self.model(state, curr_node=curr_node)
        
        action_dist = torch.distributions.Categorical(policy_mask)
        action_index = action_dist.sample()
        action = state.ndata['nid'][action_index].item()
        old_log_prob = action_dist.log_prob(action_index).detach()
        
        return action, old_log_prob, policy_mask, value_mask

    def optimize(self, ep_states, ep_actions, ep_log_probs, ep_reward_rtg, curr_node_list):
        
        batch_state = dgl.batch(ep_states)
        batch_action = torch.tensor(ep_actions)
        batch_old_log_prob = torch.tensor(ep_log_probs)
        batch_reward_rtg = torch.tensor(ep_reward_rtg)
        
        batch_old_value = self.model.critic_forward(batch_state)
        advantage = batch_reward_rtg - batch_old_value.detach()
        
        # normalization the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10) 
        
        log.info(batch_action.shape, batch_old_log_prob.shape, batch_old_value.shape, batch_reward_rtg.shape)
        log.info("batch_state.ndata['nid']", batch_state.ndata['nid'])
        log.info("curr_node_list", curr_node_list)
                
        for _ in range(2):
            # Update model
            self.optimizer.zero_grad()
            
            # shuffle the dataset
            indices = torch.randperm(len(ep_states))

            # Compute ratio
            policy_batch = self.model.actor_forward(ep_states, curr_node_list=curr_node_list)[indices]
            
            log.info(len(ep_states), len(ep_actions), len(ep_log_probs), len(ep_reward_rtg))
            log.info(self.model.critic_forward(batch_state), indices)
            value_batch = self.model.critic_forward(batch_state)[indices]
            
            action_dist_batch = torch.distributions.Categorical(policy_batch)
            
            log.info("policy_batch", policy_batch)
            log.info("batch_action", batch_action)
            log.info("ep_states[0].ndata['nid']", ep_states[0].ndata['nid'])
            log.info("torch where", )
            batch_action_index = [torch.where(state.ndata['nid']==action)[0].item() for state, action in zip(ep_states, batch_action)]
            batch_action_index = torch.tensor(batch_action_index)[indices]
            
            # TODO: 
            # (done) from batch_action go back to index
            log_prob_batch = action_dist_batch.log_prob(batch_action_index)
            
            ratio = torch.exp(log_prob_batch - batch_old_log_prob[indices])
            
            # PPO loss
            surrogate1 = ratio * advantage[indices]
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[indices]
            actor_loss = (-torch.min(surrogate1, surrogate2)).mean()
            critic_loss = torch.mean(torch.abs(value_batch - batch_reward_rtg[indices]))

            entropy_loss = action_dist_batch.entropy().mean()
            total_loss = actor_loss + critic_loss# - 0.05*entropy_loss
            
            total_loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        return total_loss.detach().item(), actor_loss.detach().item(), critic_loss.detach().item(), entropy_loss.detach().item()

class PPO_GNN_new:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        try:
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            #self.scheduler = MultiStepLR(self.optimizer, milestones=[3000, 6000], gamma=0.5)
            self.scheduler = StepLR(self.optimizer, step_size = 4000, gamma=0.1)
            self.gamma = self.args.gamma # Discount factor
        except:
            pass
        
        self.loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.clip_epsilon = 0.2
       
    def select_action(self, state, curr_node):
        policy_mask = self.model.actor_forward(state, curr_node=curr_node)
        value_mask = self.model.critic_forward(state)
        # policy_mask, value_mask = self.model(state, curr_node=curr_node)
        
        action_dist = torch.distributions.Categorical(policy_mask)
        action_index = action_dist.sample()
        action = state.ndata['nid'][action_index].item()
        old_log_prob = action_dist.log_prob(action_index).detach()
        
        return action, old_log_prob, policy_mask, value_mask

    def optimize(self, ep_states, ep_actions, ep_log_probs, ep_reward_rtg, curr_node_list):
        
        batch_state = dgl.batch(ep_states)
        batch_action = torch.tensor(ep_actions)
        batch_old_log_prob = torch.tensor(ep_log_probs)
        batch_reward_rtg = torch.tensor(ep_reward_rtg)
        
        batch_old_value = self.model.critic_forward(batch_state)
        advantage = batch_reward_rtg - batch_old_value.detach()
        
        # normalization the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10) 
        
        log.info(batch_action.shape, batch_old_log_prob.shape, batch_old_value.shape, batch_reward_rtg.shape)
        log.info("batch_state.ndata['nid']", batch_state.ndata['nid'])
        log.info("curr_node_list", curr_node_list)
                
        for _ in range(2):
            # Update model
            self.optimizer.zero_grad()
            
            # shuffle the dataset
            indices = torch.randperm(len(ep_states))

            # Compute ratio
            policy_batch = self.model.actor_forward(ep_states, curr_node_list=curr_node_list)[indices]
            
            log.info(len(ep_states), len(ep_actions), len(ep_log_probs), len(ep_reward_rtg))
            log.info(self.model.critic_forward(batch_state), indices)
            value_batch = self.model.critic_forward(batch_state)[indices]
            
            action_dist_batch = torch.distributions.Categorical(policy_batch)
            
            log.info("policy_batch", policy_batch)
            log.info("batch_action", batch_action)
            log.info("ep_states[0].ndata['nid']", ep_states[0].ndata['nid'])
            log.info("torch where", )
            batch_action_index = [torch.where(state.ndata['nid']==action)[0].item() for state, action in zip(ep_states, batch_action)]
            batch_action_index = torch.tensor(batch_action_index)[indices]
            
            # TODO: 
            # (done) from batch_action go back to index
            log_prob_batch = action_dist_batch.log_prob(batch_action_index)
            
            ratio = torch.exp(log_prob_batch - batch_old_log_prob[indices])
            
            # PPO loss
            surrogate1 = ratio * advantage[indices]
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[indices]
            
            actor_loss = (-torch.min(surrogate1, surrogate2)).mean()

            critic_loss = torch.mean(torch.abs(value_batch - batch_reward_rtg[indices]))

            entropy_loss = action_dist_batch.entropy().mean()
            total_loss = actor_loss + critic_loss# - 0.05*entropy_loss
            
            total_loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        return total_loss.detach().item(), actor_loss.detach().item(), critic_loss.detach().item(), entropy_loss.detach().item()


class PPO_MLP:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        try:
            self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)
            # self.scheduler = MultiStepLR(self.optimizer, milestones=[3000, 6000], gamma=0.5)
            self.scheduler = StepLR(self.optimizer, step_size = 4000, gamma=0.1)
            self.gamma = self.args.gamma # Discount factor
        except:
            pass
        
        self.loss = torch.nn.HuberLoss(reduction='mean', delta=1.0)
        self.clip_epsilon = 0.2
       
    def select_action(self, state, curr_node):
        policy_mask = self.model.actor_forward(state, curr_node=curr_node)
        value_mask = self.model.critic_forward(state)
        # policy_mask, value_mask = self.model(state, curr_node=curr_node)
        
        action_dist = torch.distributions.Categorical(policy_mask)
        action_index = action_dist.sample()
        action = state.ndata['nid'][action_index].item()
        old_log_prob = action_dist.log_prob(action_index).detach()
        
        return action, old_log_prob, policy_mask, value_mask

    def optimize(self, ep_states, ep_actions, ep_log_probs, ep_reward_rtg, curr_node_list):
        
        batch_state = dgl.batch(ep_states)
        batch_action = torch.tensor(ep_actions)
        batch_old_log_prob = torch.tensor(ep_log_probs)
        batch_reward_rtg = torch.tensor(ep_reward_rtg)
        
        batch_old_value = self.model.critic_forward(batch_state)
        advantage = batch_reward_rtg - batch_old_value.detach()
        
        # normalization the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10) 
        
        log.info(batch_action.shape, batch_old_log_prob.shape, batch_old_value.shape, batch_reward_rtg.shape)
        log.info("batch_state.ndata['nid']", batch_state.ndata['nid'])
        log.info("curr_node_list", curr_node_list)
                
        for _ in range(2):
            # Update model
            self.optimizer.zero_grad()
            
            # shuffle the dataset
            indices = torch.randperm(len(ep_states))

            # Compute ratio
            policy_batch = self.model.actor_forward(ep_states, curr_node_list=curr_node_list)[indices]
            
            log.info(len(ep_states), len(ep_actions), len(ep_log_probs), len(ep_reward_rtg))
            log.info(self.model.critic_forward(batch_state), indices)
            value_batch = self.model.critic_forward(batch_state)[indices]
            
            action_dist_batch = torch.distributions.Categorical(policy_batch)
            
            log.info("policy_batch", policy_batch)
            log.info("batch_action", batch_action)
            log.info("ep_states[0].ndata['nid']", ep_states[0].ndata['nid'])
            log.info("torch where", )
            batch_action_index = [torch.where(state.ndata['nid']==action)[0].item() for state, action in zip(ep_states, batch_action)]
            batch_action_index = torch.tensor(batch_action_index)[indices]
            
            # TODO: 
            # (done) from batch_action go back to index
            log_prob_batch = action_dist_batch.log_prob(batch_action_index)
            
            ratio = torch.exp(log_prob_batch - batch_old_log_prob[indices])
            
            # PPO loss
            surrogate1 = ratio * advantage[indices]
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage[indices]
            actor_loss = (-torch.min(surrogate1, surrogate2)).mean()

            critic_loss = torch.mean(torch.abs(value_batch - batch_reward_rtg[indices]))

            entropy_loss = action_dist_batch.entropy().mean()
            
            total_loss = actor_loss + critic_loss# - 0.05*entropy_loss
            
            total_loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
        return total_loss.detach().item(), actor_loss.detach().item(), critic_loss.detach().item(), entropy_loss.detach().item()

