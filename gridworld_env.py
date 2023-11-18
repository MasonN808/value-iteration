import math
import numpy as np
import random

class GridWorld():
    
    def __init__(self):
        # (x-axis, y-axis)
        self.start_state = (0, 0) # Top left corner
        self.current_state = self.start_state
        self.goal_state = [(4, 4)]  # Bottom right corner
        self.grid = [[0 for _ in range(5)] for _ in range(5)]  # Initialize gridworld

        # Define obstacles and landmarks
        self.obstacles = [(2, 2), (3, 2)]
        self.water = [(4, 2)]
        self.gold = []

        # Define action outcomes
        self.actions = {
            'UP': (-1, 0),
            'DOWN': (1, 0),
            'LEFT': (0, -1),
            'RIGHT': (0, 1)
        }

        self.dynamics = {
            'intended': 0.8,
            'right': 0.05,
            'left': 0.05,
            'none': 0.1
        }

    def reward(self, next_state: tuple):
        if next_state in self.water:
            return -10
        if next_state in self.goal_state and next_state not in self.gold:
            return 10
        if next_state in self.gold:
            return 4.5
        return 0
    
    def transition_prob(self, current_state: tuple, action: str, next_state: tuple):
        # Check current or next state is not in obstacle
        if current_state in self.obstacles or next_state in self.obstacles:
            return 0
        
        # Calculate the resulting state for each possible action outcome
        intended_result = (current_state[0] + self.actions[action][0], current_state[1] + self.actions[action][1])
        right_result = (current_state[0] + self.actions['RIGHT'][0], current_state[1] + self.actions['RIGHT'][1])
        left_result = (current_state[0] + self.actions['LEFT'][0], current_state[1] + self.actions['LEFT'][1])
        no_move_result = current_state
        
        # Check boundaries and obstacles
        if not (0 <= intended_result[0] < 5 and 0 <= intended_result[1] < 5) or intended_result in self.obstacles:
            intended_result = current_state
        if not (0 <= right_result[0] < 5 and 0 <= right_result[1] < 5) or right_result in self.obstacles:
            right_result = current_state
        if not (0 <= left_result[0] < 5 and 0 <= left_result[1] < 5) or left_result in self.obstacles:
            left_result = current_state
        
        # Calculate the transition probability
        prob = 0.0
        if next_state == intended_result:
            prob += self.dynamics['intended']
        if next_state == right_result:
            prob += self.dynamics['right']
        if next_state == left_result:
            prob += self.dynamics['left']
        if next_state == no_move_result:
            prob += self.dynamics['none']
            # print(f'PROB: {prob}')
    
        return prob