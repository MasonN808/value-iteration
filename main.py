import json
import pickle

from gridworld_env import GridWorld
import numpy as np
import math
import matplotlib.pyplot as plt

class ValueIteration():
    def __init__(self):
        self.gamma = .9
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
            iterations += 1
            self.policy_eval()
            optimal = self.policy_improv()
    
    def policy_eval(self):
        # Reset value functions
        self.value_grid = [[0 for _ in range(5)] for _ in range(5)]
        # Adding arbitrary constant to enter while loop
        Delta = self.theta + 1
        while Delta < self.theta:
            Delta = 0
            # Loop through all possible state locations
            for state in [(i, j) for i in range(5) for j in range(5)]:
                # Get current state-value
                v = self.value_grid[state[0]][state[1]]
                v_update = sum(self.policy_prob(state, action) * sum(self.env.transition_prob(state, action, next_state) * (self.env.reward(next_state) + self.gamma * self.value_grid[next_state[0]][next_state[1]]) for next_state in self.possible_states) for action in self.actions)
                self.value_grid[state[0]][state[1]] = v_update
                Delta = max(Delta, math.abs(v - v_update))

    def policy_improv(self):
        equal_policies = True
        # Loop through all possible state locations
        for state in [(i, j) for i in range(5) for j in range(5)]:
            prev_policy = self.policy[state]
            self.policy[state] = self.argmax_action(state)
            if self.policy[state] != prev_policy:
                equal_policies = False

        if equal_policies:
            return True
        return False
        
    def policy_prob(self, state: tuple, action: str):
        if action != self.policy(state):
            return 0
        return 1
    
    def argmax_action(self, state):
        action_values = {}

        for action in self.actions:
            action_value = sum(
                self.policy_prob(state, action) * sum(
                    self.env.transition_prob(state, action, next_state) *
                    (self.env.reward(next_state) + self.gamma * self.value_grid[next_state[0]][next_state[1]])
                    for next_state in self.possible_states
                )
            )
            action_values[action] = action_value

        # Find the action with the maximum expected return
        best_action = max(action_values, key=action_values.get)
        return best_action


if __name__ == "__main__":
    # TODO visualize policy and value function