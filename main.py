import copy
import json
import pickle
import random

from gridworld_env import GridWorld
import numpy as np
import math
import matplotlib.pyplot as plt

class ModelFree():
    def __init__(self):
        self.gamma = .9
        self.theta = .0001 # Stopping condition for policy evaluation
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)] # matrix of state-value functions
        self.value_grid_new = [[0 for _ in range(5)] for _ in range(5)] # matrix of state-value functions
        self.env = GridWorld()
        self.possible_states = [(i, j) for i in range(5) for j in range(5) if (i, j) not in self.env.obstacles]
        self.policy = {}
        # Use a for loop to generate keys and assign the default value "RIGHT" to each key
        for i in range(5):
            for j in range(5):
                self.policy[(i, j)] = "RIGHT"

        # Define actions
        self.actions = [
            'UP',
            'DOWN',
            'LEFT',
            'RIGHT']

    def value_iter(self):
        # Reset value functions
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)]
        self.value_grid_new = [[0 for _ in range(5)] for _ in range(5)]
        # Adding arbitrary constant to enter while loop
        iterations = 0
        Delta = self.theta + 1
        while Delta > self.theta:
            # self.visualize_vf()
            print(iterations)
            iterations += 1
            Delta = 0
            # Loop through all possible state locations
            for state in self.possible_states:
                # Get current state-value
                v = self.value_grid[state[0]][state[1]]
                action_values = []
                for action in self.actions:
                    transition_loop = 0
                    for next_state in self.possible_states:
                        # Ensure that any transistion from the goal state results in 0 reward
                        if state in self.env.goal_state or next_state in self.env.obstacles or state in self.env.obstacles:
                            reward = 0
                            next_value = 0
                        else:
                            reward = self.env.reward(next_state)
                            next_value = self.value_grid[next_state[0]][next_state[1]]
                        transition_loop += self.env.transition_prob(state, action, next_state) * (reward + self.gamma * next_value)
                        # if state == (4, 1) and next_state == (4, 2):
                            # print(self.env.transition_prob(state, action, next_state))
                    action_value = transition_loop
                    action_values.append(action_value)
                max_value = max(action_values)
                self.value_grid_new[state[0]][state[1]] = max_value
                Delta = max(Delta, abs(v - max_value))
            self.value_grid = self.value_grid_new

        for state in self.possible_states:
            self.policy[state] = self.argmax_action(state)
        
        return self.policy, self.value_grid
    
    def argmax_action(self, state):
        action_values = {}
        for action in self.actions:
            transition_loop = 0
            for next_state in self.possible_states:
                # Ensure that any transistion from the goal state results in 0 reward
                if state in self.env.goal_state or next_state in self.env.obstacles or state in self.env.obstacles:
                    reward = 0
                    next_value = 0
                else:
                    reward = self.env.reward(next_state)
                    next_value = self.value_grid[next_state[0]][next_state[1]]
                transition_loop += self.env.transition_prob(state, action, next_state) * (reward + self.gamma * next_value)
            action_value = transition_loop

            action_values[action] = action_value

        # Find the action with the maximum expected return
        best_action = max(action_values, key=action_values.get)
        # if state in self.env.gold:
        #     print(action_values)
        return best_action
    
    def td_learning(self, policy: dict, alpha: float, delta: float, gamma: float, num_iter: int):
        # Reset value functions
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)]
        self.value_grid_new = [[0 for _ in range(5)] for _ in range(5)]
        self.average_value_grid = [[0 for _ in range(5)] for _ in range(5)]
        assert delta > 0, "delta should be greater than 0'"
        episodes = []
        for n in range(0, num_iter):
            max_norm = delta + .001
            episode_num = 0
            while max_norm > delta:
                max_norm = 0
                episode_num +=1
                # Initialize start state
                state = random.choice(self.possible_states)
                # Generate a trajectory using the policy
                while state not in self.env.goal_state:
                    # Get current state-value
                    action = policy[state]
                    reward, next_state = self.env.step(state, action)
                    v_prev = self.value_grid[state[0]][state[1]]
                    v_next = self.value_grid[next_state[0]][next_state[1]]
                    v = v_prev + alpha*(reward + gamma*v_next - v_prev)
                    if max_norm < abs(v - v_prev):
                        max_norm = abs(v - v_prev)
                    self.value_grid_new[state[0]][state[1]] = v
                    state = next_state
                self.value_grid = self.value_grid_new
            
            episodes.append(episode_num)
            for state in self.possible_states:
                self.average_value_grid[state[0]][state[1]] += self.value_grid[state[0]][state[1]]

        for state in self.possible_states:
                self.average_value_grid[state[0]][state[1]] /= n

        return self.average_value_grid, episodes
    
    def calculate_max_norm(self, vf1, vf2): 
        max_norm = 0
        for state in self.possible_states:
            norm = abs(vf1[state[0]][state[1]] - vf2[state[0]][state[1]])
            if max_norm < norm:
                max_norm = norm
        return max_norm

    def visualize_vf(self, value_grid=None):
        if not value_grid:
            value_grid = self.value_grid

        # Set up the figure
        _, ax = plt.subplots(figsize=(8, 8))

        # Set the grid
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.grid(which='both')

        # Set the ticks
        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 5, 1))

        # Set aspect of the plot to be equal
        ax.set_aspect('equal')

        # Add values
        for i in range(5):
            for j in range(5):
                ax.text(j + 0.5, 4.5 - i, f'{value_grid[i][j]:.4f}', ha="center", va="center")

        # Show the plot
        plt.show()
    
    def visualize_policy(self):
        # Set up the figure
        _, ax = plt.subplots(figsize=(8, 8))

        # Set the grid
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.grid(which='both')

        # Set the ticks
        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 5, 1))
        ax.set_aspect('equal')

        # Loop over the states and create an arrow for each policy direction
        for state, action in self.policy.items():
            # Starting point of the arrow
            start_point = (state[1] + 0.5, 4.5 - state[0])
            if state in self.env.goal_state:
                ax.text(state[1] + 0.5, 4.5 - state[0], 'G', ha="center", va="center", fontsize=12)
                continue

            # Direction of the arrow
            if action == 'UP':
                dx, dy = 0, -0.4
            elif action == 'DOWN':
                dx, dy = 0, 0.4
            elif action == 'LEFT':
                dx, dy = -0.4, 0
            elif action == 'RIGHT':
                dx, dy = 0.4, 0

            # Draw the arrow
            ax.arrow(start_point[0], start_point[1], dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

        # Show the plot
        plt.show()


if __name__ == "__main__":
    mf = ModelFree()
    optimal_policy, optimal_value_grid = mf.value_iter()
    averaged_value_grid, episodes = mf.td_learning(policy=optimal_policy, alpha=.35, delta=.0001, gamma=.9, num_iter=50)
    print(mf.calculate_max_norm(averaged_value_grid, optimal_value_grid))
    print(np.mean(episodes))
    print(np.std(episodes))
    mf.visualize_vf(value_grid=averaged_value_grid)