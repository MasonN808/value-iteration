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
    
    def sarsa(self, alpha: float, gamma: float, epsilon: float, episodes: int, optimal_value_grid=None):
        value_grid = [[0 for _ in range(5)] for _ in range(5)]
        policy = {}
        num_actions_per_episode = []
        mean_sqr_error_per_episode = []
        # Reset Q-function
        q_values = {}
        for i in range(5):
            for j in range(5):
                for action in self.actions:
                    q_values[((i, j), action)] = 1
        num_actions = 0
        for _ in range(0, episodes):
            state = self.env.start_state
            action = self.epsilon_greedy_policy(state=state, epsilon=epsilon, q_values=q_values)
            # Repeat for each step of episode
            while state not in self.env.goal_state:
                num_actions += 1
                reward, next_state = self.env.step(state, action)
                next_action = self.epsilon_greedy_policy(state=next_state, epsilon=epsilon, q_values=q_values)
                q_value = q_values[((state[0], state[1]), action)]
                q_value_next = q_values[((next_state[0], next_state[1]), next_action)]
                q_values[((state[0], state[1]), action)] = q_value + alpha * (reward + gamma*q_value_next - q_value)
                state = next_state
                action = next_action

            # Calculate mean square error
            if optimal_value_grid:
                error_squared_list = []
                for state in self.possible_states:
                    value_function = sum([self.epsilon_greedy_policy_prob(state, action, epsilon, q_values)*q_values[((state[0], state[1]), action)] for action in self.actions])
                    # value_grid[state[0]][state[1]] = value_function
                    value_grid[state[0]][state[1]] = max([q_values[((state[0], state[1]), action)] for action in self.actions])
                    optimal_value_funtion = optimal_value_grid[state[0]][state[1]]
                    error_squared_list.append((value_function - optimal_value_funtion)**2)
                mean_sqr_error = sum(error_squared_list)/len(self.possible_states)
                mean_sqr_error_per_episode.append(mean_sqr_error)

            num_actions_per_episode.append(num_actions)
        return num_actions_per_episode, mean_sqr_error_per_episode, value_grid
    
    def q_learning(self, alpha: float, gamma: float, epsilon: float, episodes: int, optimal_value_grid=None):
        value_grid = [[0 for _ in range(5)] for _ in range(5)]
        policy = {}
        num_actions_per_episode = []
        mean_sqr_error_per_episode = []
        # Reset Q-function
        q_values = {}
        for i in range(5):
            for j in range(5):
                for action in self.actions:
                    q_values[((i, j), action)] = 1
        num_actions = 0
        for i in range(0, episodes):
            state = self.env.start_state
            # Repeat for each step of episode
            print(i)
            while state not in self.env.goal_state:
                action = self.epsilon_greedy_policy(state=state, epsilon=epsilon, q_values=q_values)
                num_actions += 1
                reward, next_state = self.env.step(state, action)
                q_value = q_values[((state[0], state[1]), action)]
                q_value_max = max([q_values[((next_state[0], next_state[1]), action)] for action in self.actions])
                q_values[((state[0], state[1]), action)] = q_value + alpha * (reward + gamma*q_value_max - q_value)
                state = next_state

            # Calculate mean square error
            if optimal_value_grid:
                error_squared_list = []
                for state in self.possible_states:
                    value_function = sum([self.epsilon_greedy_policy_prob(state, action, epsilon, q_values)*q_values[((state[0], state[1]), action)] for action in self.actions])
                    value_grid[state[0]][state[1]] = max([q_values[((state[0], state[1]), action)] for action in self.actions])
                    optimal_value_funtion = optimal_value_grid[state[0]][state[1]]
                    error_squared_list.append((value_function - optimal_value_funtion)**2)
                mean_sqr_error = sum(error_squared_list)/len(self.possible_states)
                mean_sqr_error_per_episode.append(mean_sqr_error)

            num_actions_per_episode.append(num_actions)
        return num_actions_per_episode, mean_sqr_error_per_episode, value_grid

    def epsilon_greedy_policy(self, state: tuple, epsilon: float, q_values: dict):
        action_values = self.get_values_for_key_prefix(dictionary=q_values, key_prefix=state)
        if epsilon < random.uniform(0,1):
            # Get the action value that maximizes value
            action_value = max(action_values, key=lambda x: x[1])
        else:
            action_value = random.choice(action_values)
        return action_value[0]
    
    def epsilon_greedy_policy_prob(self, state: tuple, action: str, epsilon: float, q_values: dict):
        action_values = self.get_values_for_key_prefix(dictionary=q_values, key_prefix=state)
        # Find the maximum value
        max_value = max(action_values, key=lambda x: x[1])[1]
        # Get all actions that have this maximum value
        max_actions = [action for action, value in action_values if value == max_value]
        # Check if this is the action that maximizes return
        if action in max_actions:
            return (1-epsilon)/len(max_actions) + epsilon/len(self.actions)
        else:
            return epsilon/len(self.actions)

    @staticmethod
    def get_values_for_key_prefix(dictionary: dict, key_prefix: tuple):
        """
        Function to extract values from a dictionary where keys are tuples,
        and the first element of the key tuple matches the given key_prefix.
        """
        return [(key[1], value) for key, value in dictionary.items() if key[0] == key_prefix]

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
    
    def visualize_policy(self, policy: dict):
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
        for state, action in policy.items():
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
    # TD-Learning
    # mf = ModelFree()
    # optimal_policy, optimal_value_grid = mf.value_iter()
    # averaged_value_grid, episodes = mf.td_learning(policy=optimal_policy, alpha=.35, delta=.0001, gamma=.9, num_iter=50)
    # print(mf.calculate_max_norm(averaged_value_grid, optimal_value_grid))
    # print(np.mean(episodes))
    # print(np.std(episodes))
    # mf.visualize_vf(value_grid=averaged_value_grid)

    # SARSA Q2a)
    # mf = ModelFree()
    # num_iter = 20
    # alpha = .5
    # epsilon = .01
    # num_actions_agg, _, _ = mf.sarsa(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170)
    # for _ in range(num_iter-1):
    #     num_actions, _, _ = mf.sarsa(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170)
    #     num_actions_agg = [sum(x) for x in zip(num_actions, num_actions_agg)]
    # num_actions_agg = [x / num_iter for x in num_actions_agg]

    # # Learning Curve
    # # Number of ticks you want to display
    # num_ticks = 10
    # # Generate linearly spaced x-ticks
    # min_val = min(num_actions_agg)
    # max_val = max(num_actions_agg)
    # linear_ticks = np.linspace(min_val, max_val, num_ticks)

    # plt.figure(figsize=(8, 4))
    # plt.plot(num_actions_agg, range(len(num_actions_agg)), marker='.')  # Plotting index vs elements
    # plt.xlabel('Time steps')
    # plt.ylabel('Episodes')
    # plt.xticks(range(0, int(max(num_actions_agg)) + 1000, 1000))
    # # plt.xticks(rotation=45)
    # plt.grid(True)
    # plt.show()

    # SARSA Q2b) and 2c)
    mf = ModelFree()
    # Get optimal value function for error logs
    _, optimal_value_grid = mf.value_iter()
    num_iter = 20
    alpha = .5
    epsilon = .01
    agg_value_grid = [[0 for _ in range(5)] for _ in range(5)]
    _, mean_sqr_error_agg, value_grid = mf.sarsa(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170, optimal_value_grid=optimal_value_grid)
    for state in mf.possible_states:
        agg_value_grid[state[0]][state[1]] += value_grid[state[0]][state[1]]
    for _ in range(num_iter-1):
        _, mean_sqr_error, value_grid = mf.sarsa(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170, optimal_value_grid=optimal_value_grid)
        mean_sqr_error_agg = [sum(x) for x in zip(mean_sqr_error, mean_sqr_error_agg)]
        for state in mf.possible_states:
            agg_value_grid[state[0]][state[1]] += value_grid[state[0]][state[1]]

    for state in mf.possible_states:
        agg_value_grid[state[0]][state[1]] /= num_iter
    # Generate the policy from the avergaged value function derived from the Q-values
    policy = {}
    for state in mf.possible_states:
        policy[state] = mf.argmax_action(state)

    mf.visualize_policy(policy)
    mean_sqr_error_agg = [x / num_iter for x in mean_sqr_error_agg]

    # Learning Curve
    # Number of ticks you want to display
    num_ticks = 10
    # Generate linearly spaced x-ticks
    min_val = min(mean_sqr_error_agg)
    max_val = max(mean_sqr_error_agg)
    linear_ticks = np.linspace(min_val, max_val, num_ticks)

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(mean_sqr_error_agg)), mean_sqr_error_agg, marker='.')  # Plotting index vs elements
    plt.xlabel('Episodes')
    plt.ylabel('Mean Squared-Error')
    plt.yticks(range(0, int(max(mean_sqr_error_agg)), 2))
    plt.grid(True)
    plt.show()

    # Q-Learning Q3a)
    # mf = ModelFree()
    # num_iter = 20
    # alpha = .5
    # epsilon = .01
    # num_actions_agg, _, _ = mf.q_learning(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170)
    # for _ in range(num_iter-1):
    #     num_actions, _, _ = mf.q_learning(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170)
    #     num_actions_agg = [sum(x) for x in zip(num_actions, num_actions_agg)]
    # num_actions_agg = [x / num_iter for x in num_actions_agg]

    # # Learning Curve
    # # Number of ticks you want to display
    # num_ticks = 10
    # # Generate linearly spaced x-ticks
    # min_val = min(num_actions_agg)
    # max_val = max(num_actions_agg)
    # linear_ticks = np.linspace(min_val, max_val, num_ticks)

    # plt.figure(figsize=(8, 4))
    # plt.plot(num_actions_agg, range(len(num_actions_agg)), marker='.')  # Plotting index vs elements
    # plt.xlabel('Time steps')
    # plt.ylabel('Episodes')
    # plt.xticks(range(0, int(max(num_actions_agg)) + 1000, 1000))
    # plt.grid(True)
    # plt.show()

    # Q-Learning Q3b) and 3c)
    # mf = ModelFree()
    # # Get optimal value function for error logs
    # _, optimal_value_grid = mf.value_iter()
    # num_iter = 20
    # alpha = .5
    # epsilon = .01
    # agg_value_grid = [[0 for _ in range(5)] for _ in range(5)]
    # _, mean_sqr_error_agg, value_grid = mf.q_learning(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170, optimal_value_grid=optimal_value_grid)
    # for state in mf.possible_states:
    #     agg_value_grid[state[0]][state[1]] += value_grid[state[0]][state[1]]
    # for _ in range(num_iter-1):
    #     _, mean_sqr_error, value_grid = mf.q_learning(alpha=alpha, gamma=.9, epsilon=epsilon, episodes=170, optimal_value_grid=optimal_value_grid)
    #     mean_sqr_error_agg = [sum(x) for x in zip(mean_sqr_error, mean_sqr_error_agg)]
    #     for state in mf.possible_states:
    #         agg_value_grid[state[0]][state[1]] += value_grid[state[0]][state[1]]

    # for state in mf.possible_states:
    #     agg_value_grid[state[0]][state[1]] /= num_iter
    # # Generate the policy from the avergaged value function derived from the Q-values
    # policy = {}
    # for state in mf.possible_states:
    #     policy[state] = mf.argmax_action(state)

    # mf.visualize_policy(policy)
    # mean_sqr_error_agg = [x / num_iter for x in mean_sqr_error_agg]

    # # Learning Curve
    # # Number of ticks you want to display
    # num_ticks = 10
    # # Generate linearly spaced x-ticks
    # min_val = min(mean_sqr_error_agg)
    # max_val = max(mean_sqr_error_agg)
    # linear_ticks = np.linspace(min_val, max_val, num_ticks)

    # plt.figure(figsize=(8, 4))
    # plt.plot(range(len(mean_sqr_error_agg)), mean_sqr_error_agg, marker='.')  # Plotting index vs elements
    # plt.xlabel('Episodes')
    # plt.ylabel('Mean Squared-Error')
    # plt.yticks(range(0, int(max(mean_sqr_error_agg)), 2))
    # plt.grid(True)
    # plt.show()
    