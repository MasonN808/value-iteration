import json
import pickle

from gridworld_env import GridWorld
import numpy as np
import math
import matplotlib.pyplot as plt

class ValueIteration():
    def __init__(self):
        self.gamma = .25
        self.theta = .0001 # Stopping condition for policy evaluation
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)] # matrix of state-value functions
        self.possible_states = [(i, j) for i in range(5) for j in range(5)]
        self.env = GridWorld()
        # Initialize an empty dictionary
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
        optimal = False
        iterations = 0
        while not optimal:
            print(iterations)
            iterations += 1
            self.policy_eval()
            optimal = self.policy_improv()
    
    def policy_eval(self):
        # Reset value functions
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)]
        # Adding arbitrary constant to enter while loop
        Delta = self.theta + 1
        while Delta > self.theta:
            Delta = 0
            # Loop through all possible state locations
            for state in self.possible_states:
                # Get current state-value
                v = self.value_grid[state[0]][state[1]]
                v_update = 0
                for action in self.actions:
                    transition_loop = 0
                    for next_state in self.possible_states:
                        # Ensure that any transistion from the goal state results in 0 reward
                        if state == self.env.goal_state:
                            reward = 0
                            next_value = 0
                        else:
                            reward = self.env.reward(next_state)
                            next_value = self.value_grid[next_state[0]][next_state[1]]
                        transition_loop += self.env.transition_prob(state, action, next_state) * (reward + self.gamma * next_value)
                    v_update += self.policy_prob(state, action) * transition_loop
                self.value_grid[state[0]][state[1]] = v_update
                Delta = max(Delta, abs(v - v_update))

    def policy_improv(self) -> bool:
        equal_policies = True
        # Loop through all possible state locations
        for state in self.possible_states:
            prev_policy = self.policy[state]
            self.policy[state] = self.argmax_action(state)
            if self.policy[state] != prev_policy:
                equal_policies = False
        return equal_policies
        
    def policy_prob(self, state: tuple, action: str):
        if action != self.policy.get(state):
            return 0
        return 1
    
    def argmax_action(self, state):
        action_values = {}
        for action in self.actions:
            transition_loop = 0
            for next_state in self.possible_states:
                # Ensure that any transistion from the goal state results in 0 reward
                if state == self.env.goal_state:
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
        return best_action
    
    def visualize_vf(self):
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
                ax.text(j + 0.5, 4.5 - i, f'{self.value_grid[i][j]:.4f}', ha="center", va="center")

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
            if state == self.env.goal_state:
                ax.text(state[1] + 0.5, 4.5 - state[0], 'G', ha="center", va="center", fontsize=12)
                continue

            # Direction of the arrow
            if action == 'UP':
                dx, dy = 0, 0.4
            elif action == 'DOWN':
                dx, dy = 0, -0.4
            elif action == 'LEFT':
                dx, dy = -0.4, 0
            elif action == 'RIGHT':
                dx, dy = 0.4, 0

            # Draw the arrow
            ax.arrow(start_point[0], start_point[1], dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')

        # Show the plot
        plt.show()


if __name__ == "__main__":
    vi = ValueIteration()
    vi.value_iter()
    vi.visualize_policy()
    vi.visualize_vf()