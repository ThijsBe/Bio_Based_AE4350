# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:16:00 2023

@author: thijs
"""
#import the libraries
import numpy as np 
import matplotlib.pyplot as plt
import gym
import random
from visualize import Animation
from pathlib import Path
from Randomized_spawns import random_spawns
import argparse
from scipy.signal import savgol_filter


#Map file location
filename = r"C:\Users\thijs\OneDrive\Bureaublad\Bio_Based\Assignment.txt"

#Import map
def import_mapf_instance(filename):
    f = Path(filename)
    if not f.is_file():
        raise BaseException(filename + " does not exist.")
    f = open(filename, 'r')
    # first line: #rows #columns
    line = f.readline()
    rows, columns = [int(x) for x in line.split(' ')]
    rows = int(rows)
    columns = int(columns)
    # #rows lines with the map
    my_map = []
    for r in range(rows):
        line = f.readline()
        my_map.append([])
        for cell in line:
            if cell == '@':
                my_map[-1].append(True)
            elif cell == '.':
                my_map[-1].append(False)
    #agents
    line = f.readline()
    num_agents = int(line)
    #agents lines with the start/goal positions
    starts = []
    goals = []
    for a in range(num_agents):
        line = f.readline()
        sx, sy, gx, gy = [int(x) for x in line.split(' ')]
        starts.append((sx, sy))
        goals.append((gx, gy))
    f.close()
    return my_map, starts, goals

######################################################################################
#Run simulations multiple times and change variables
Nums = []
Colls = []
sumrewards = []
for n in range(1,8,1):
    total_cost = []
    cpu = []
    cv_list = []
    counter = 0
    ii = 0 
    N = n #Number of agents
    
    #Pick N random start and goal locations
    random_spawns(N)
    my_map, starts, goals = import_mapf_instance(filename)
    
    class CustomGridEnv(gym.Env):
        def __init__(self, starts, goals, n, m):
            self.grid_shape = (n , m)
            self.start_pos = starts
            self.target_pos = goals
            self.current_pos = self.start_pos
    
            #Action space: 0 - up, 1 - down, 2 - left, 3 - right, 4 - stay
            self.action_space = gym.spaces.Discrete(5)
    
            #Observation space: (current_row, current_col)
            self.observation_space = gym.spaces.Tuple(
                (gym.spaces.Discrete(self.grid_shape[0]), gym.spaces.Discrete(self.grid_shape[1]))
            )
    
        def reset(self):
            self.current_pos = self.start_pos
            return self.current_pos
        
        def step(self, action, state_dict, N, i):
            state_d = 0
            row, col = self.current_pos
            
            row_prev, col_prev = self.current_pos
            
            #Agent is not allowed to move when goal is reached to speed up computations
            if self.current_pos == self.target_pos:
                action = 4
    
            if action == 0:  #Up
                row = max(0, row - 1)
           
            elif action == 1:  #Down
                row = min(self.grid_shape[0] - 1, row + 1)
         
            elif action == 2:  #Left
                col = max(0, col - 1)
           
            elif action == 3:  #Right
                col = min(self.grid_shape[1] - 1, col + 1)
            
            elif action == 4:  #Stay
                row, col = self.current_pos
            
            self.current_pos = (row, col)
            reward = 0
            for j in range(N):
                if j == i:
                    continue
                
                #Dictionary containing all agent's positions
                x1 = np.array(state_dict['{}'.format(j)])
                y1 = np.array(self.current_pos)
                z1 = np.array(self.target_pos)
                #Previous position of agent
                w1 = np.array((row_prev, col_prev))
                
                
                #Punish getting too close towards other agents
                #Agents receive large negative reward when they collide
                if np.linalg.norm(x1 - y1) < 0.1:
                    reward += - 50.0
                #Agents receive rewards based on distance function
                for k in range(14):
                    if np.linalg.norm(x1 - y1) < k:
                        reward += -np.exp(-k)
                #Reward moving towards goal location
                if np.linalg.norm(z1 - y1) < np.linalg.norm(z1 - w1):
                    reward += 0.4
                else:
                    reward += -0.7
            #Agent's receive big reward when they reach goal location
            done = self.current_pos == self.target_pos
            reward_tot = 5.0 + reward if done else reward
            if done == True:
                state_d = 1
    
            return self.current_pos, reward_tot, done, state_d, state_dict, {}
    
    rewards = {}
    for j in range(N):
        rewards['{}'.format(j)] = []

    def train_agents(env, N, num_episodes=1, learning_rate= 0.99, discount_factor= 0.99, exploration_rate=1.0, exploration_decay=0.99, min_exploration_rate=0.01):
        q_table = {}
        state_dict = {}
        Num_Steps = []
        #Create dictionary of q tables for each agent
        for j in range(N):
            q_table['{}'.format(j)] = np.zeros((env[j].grid_shape[0], env[j].grid_shape[1], env[j].action_space.n))
        collisions = []
        sum_of_rewards = []
        jj = 0
        for episode in range(num_episodes):
            jj += 1
            print(f'Percentage done:', ((jj)/(num_episodes))*100)
            for j in range(N):
                state_dict['{}'.format(j)] = env[j].reset()
            done = False
            for j in range(N):
                states = {}
                for j in range(N):
                   states['{}'.format(j)] = []
                paths = {}
                for j in range(N):
                   paths['{}'.format(j)] = []
            steps = []
            Sum = 0
            collision = 0
            srewards = 0
            #Continue choosing actions until all agents reached their goal loc
            while Sum < N:
                #First agent 1 picks an action, then agent 2, ..., then agent N
                Sum = 0
                for i in range(N):
                    #Exploration vs Exploitation
                    if random.uniform(0, 1) < exploration_rate:
                        action = env[i].action_space.sample()
                    else:
                        action = np.argmax(q_table['{}'.format(i)][state_dict['{}'.format(i)]])
        
                    next_state, reward, done, state_d, _, _ = env[i].step(action, state_dict, N, i)
                    Sum += state_d
                    #Q-learning update
                    q_value = q_table['{}'.format(i)][state_dict['{}'.format(i)]][action]
                    max_next_q_value = np.max(q_table['{}'.format(i)][next_state])
                    new_q_value = q_value + learning_rate * (reward + discount_factor * max_next_q_value - q_value)
                    q_table['{}'.format(i)][state_dict['{}'.format(i)]][action] = new_q_value
                    rewards['{}'.format(i)].append(reward)
                    state_dict['{}'.format(i)] = next_state
                    steps.append(state_dict['{}'.format(i)])
                    
                    if episode == num_episodes - 1:
                        states['{}'.format(i)].append(state_dict['{}'.format(i)])
                    loc = []
                    for j in range(N):
                        loc.append(state_dict['{}'.format(j)])
                    flag = len(set(loc)) == len(loc)
                    if(flag):
                        None
                    else:
                        collision += 1
                    srewards += reward
    
            sum_of_rewards.append(srewards/N) #Normalize by dividing by the number of agents
            collisions.append(collision/N) #Normalize by dividing by the number of agents
            Num_Steps.append(len(steps)/N) #Normalize by dividing by the number of agents
            for j in range(N):
                paths['{}'.format(j)].append(states['{}'.format(j)])
               
    
            exploration_rate = max(exploration_rate * exploration_decay, min_exploration_rate)
        return q_table, paths, Num_Steps, collisions, sum_of_rewards
    
    if __name__ == "__main__":
        custom_env = []
        for j in range(N):
            custom_env.append(CustomGridEnv(starts[j], goals[j], len(my_map), len(my_map[0])))
        q_table, paths, Num_Steps, collisions, sum_of_rewards = train_agents(custom_env, N)
        Nums.append(Num_Steps)
        Colls.append(collisions)
        sumrewards.append(sum_of_rewards)
    #Create a list to place the paths in for animation
    paths_list = []  
    for i in range(N):
        paths_list.append(paths['{}'.format(i)][0])

#Run animation
print("***Test paths on a simulation***")
animation = Animation(my_map, starts, goals, paths_list)  
    
    ###################################################################################
    
