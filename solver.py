import math

import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, mdp):
        # mdp
        self.mdp = mdp

        self.action_value_array = np.zeros((len(self.mdp.states), len(self.mdp.actions)))
        self.episode_returns = []

        self.last_used_algorithm = ""
        self.last_used_parameters = {}

    def reset_solver(self):
        self.action_value_array = np.zeros((len(self.mdp.states), len(self.mdp.actions)))
        self.episode_returns = []
        self.last_used_algorithm = {}

    def solve_with_dynamic_programming(self, theta, gamma):
        self.last_used_algorithm = "dynamic programming"
        self.last_used_parameters = {
            "theta": theta,
            "gamma": gamma,
        }
        self.__dynamic_programming(theta, gamma)

    def solve_with_q_learning(self, episodes_enabled, policy_evaluation_enabled,
                              episodes, theta, max_steps_in_episode, start_state_index,
                              alpha, epsilon, gamma, cost_of_living):
        self.last_used_algorithm = "q learning"
        self.last_used_parameters = {
            "episodes_enabled": episodes_enabled,
            "policy_evaluation_enabled": policy_evaluation_enabled,
            "episodes": episodes,
            "theta": theta,
            "max_steps_in_episode": max_steps_in_episode,
            "start_state_index": start_state_index,
            "alpha": alpha,
            "epsilon": epsilon,
            "gamma": gamma,
            "cost_of_living": cost_of_living
        }

        if episodes_enabled:
            for episode in range(episodes):
                print(f"q_learning episode: {episode}")
                self.__q_learning(max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living)

        # policy evaluation
        episode = 0
        if policy_evaluation_enabled:
            while True:
                episode += 1
                print(f"q_learning iteration: {episode}")

                difference = self.__q_learning(max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living)
                if difference < theta:
                    break

    def solve_with_monte_carlo(self, episodes_enabled, policy_evaluation_enabled,
                               episodes, theta, max_steps_in_episode, start_state_index,
                               alpha, epsilon, gamma, cost_of_living):
        self.last_used_algorithm = "monte carlo"
        self.last_used_parameters = {
            "episodes_enabled": episodes_enabled,
            "policy_evaluation_enabled": policy_evaluation_enabled,
            "episodes": episodes,
            "theta": theta,
            "max_steps_in_episode": max_steps_in_episode,
            "start_state_index": start_state_index,
            "alpha": alpha,
            "epsilon": epsilon,
            "gamma": gamma,
            "cost_of_living": cost_of_living
        }

        if episodes_enabled:
            for episode in range(episodes):
                print(f"monte_carlo episode: {episode}")
                self.__monte_carlo(max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living)

        # policy evaluation
        episode = 0
        if policy_evaluation_enabled:
            while True:
                episode += 1
                print(f"monte_carlo iteration: {episode}")

                difference = self.__monte_carlo(max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living)
                if difference < theta:
                    break

    def __dynamic_programming(self, theta, gamma):
        # policy evaluation
        episode = 0
        while True:
            print(f"episode: {episode}")
            difference = 0
            for state in range(len(self.mdp.states)):
                for action in range(len(self.mdp.actions)):
                    old_value = self.action_value_array[state, action]
                    new_value = 0
                    for new_state in range(len(self.mdp.states)):
                        new_value += self.mdp.transition_probabilities[state, action, new_state] * (
                                self.mdp.rewards[state, action, new_state] + gamma * self.action_value_array[
                            new_state, np.argmax(self.action_value_array[new_state, :])])
                    self.action_value_array[state, action] = new_value
                    difference = max(difference, abs(old_value - new_value))
            if difference < theta:
                break

    def __q_learning(self, max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living):
        # start episode
        difference = 0
        episode_return = 0
        previous_state = start_state_index
        for step in range(max_steps_in_episode):
            best_action = np.argmax(self.action_value_array[previous_state, :])
            possible_actions = self.mdp.possible_actions_indexes(previous_state)

            # choose action based on epsilon-greedy policy
            # np.max(action_value_array[previous_state, :]) is there to balance starting states
            # when action_value_array is full of zeros and agent always goes with action with index 0
            # so agent chooses random action
            if np.random.rand() < epsilon or not np.max(
                    self.action_value_array[previous_state, :]) or best_action not in possible_actions:
                # random action
                action = np.random.choice(self.mdp.possible_actions_indexes(previous_state))
            else:
                # best action
                action = best_action

            old_value = self.action_value_array[previous_state, action]
            new_value = 0

            # make an action
            new_state, reward, is_terminal = self.mdp.step(previous_state, action)

            # take cost of living into account
            if cost_of_living != 0:
                reward += cost_of_living

            episode_return += reward

            # update Action Value function (Q)
            new_value = (1 - alpha) * self.action_value_array[
                previous_state, action] + alpha * (
                                reward + gamma * max(self.action_value_array[new_state, :]))
            self.action_value_array[previous_state, action] = new_value

            previous_state = new_state

            # if new state is terminal finish episode
            if is_terminal:
                break

            difference = max(difference, abs(old_value - new_value))

        self.episode_returns.append(episode_return)
        return difference

    def __monte_carlo(self, max_steps_in_episode, start_state_index, alpha, epsilon, gamma, cost_of_living):
        # start episode
        monte_carlo_history = []
        previous_state = start_state_index
        for step in range(max_steps_in_episode):
            best_action = np.argmax(self.action_value_array[previous_state, :])
            possible_actions = self.mdp.possible_actions_indexes(previous_state)

            # choose action based on epsilon-greedy policy
            # np.max(action_value_array[previous_state, :]) is there to balance starting states
            # when action_value_array is full of zeros and agent always goes with action with index 0
            if np.random.rand() < epsilon or not np.max(
                    self.action_value_array[previous_state, :]) or best_action not in possible_actions:
                # random action
                action = np.random.choice(self.mdp.possible_actions_indexes(previous_state))
            else:
                # best action
                action = best_action

            # make an action
            new_state, reward, is_terminal = self.mdp.step(previous_state, action)

            # take cost of living into account
            if cost_of_living != 0:
                reward += cost_of_living

            # add the episode states, actions, rewards for  Monte Carlo
            monte_carlo_history.append((previous_state, action, reward))

            previous_state = new_state

            # if new state is terminal finish episode
            if is_terminal:
                break

        # Calculate returns and update Action Value function (Q)
        difference = 0
        episode_return = 0
        for t in range(len(monte_carlo_history) - 1, -1, -1):
            state = monte_carlo_history[t][0]
            action = monte_carlo_history[t][1]
            reward = monte_carlo_history[t][2]

            old_value = self.action_value_array[state, action]
            new_value = self.action_value_array[state, action]

            episode_return = gamma * episode_return + reward
            new_value += alpha * (episode_return - self.action_value_array[state, action])
            self.action_value_array[state, action] = new_value

            difference = max(difference, abs(old_value - new_value))

        self.episode_returns.append(episode_return)
        return difference

    def create_plot_of_action_value_array(self):
        # Set the figure size
        plt.figure(figsize=(8, math.ceil(len(self.mdp.states) * 0.24)))

        # Create a heatmap
        plt.imshow(self.action_value_array, cmap='viridis', interpolation='nearest', aspect=0.2)

        # Add colorbar
        plt.colorbar()

        # Set axis labels
        plt.xlabel('Actions')
        plt.ylabel('States')

        # Set custom labels for the x-axis
        plt.xticks(np.arange(len(self.mdp.actions)), self.mdp.actions)

        # Set custom labels for the y-axis
        plt.yticks(np.arange(len(self.mdp.states)), self.mdp.states)

        # Annotate each cell with its value
        for i in range(len(self.mdp.states)):
            for j in range(len(self.mdp.actions)):
                plt.text(j, i, f'{self.action_value_array[i, j]:.2f}', ha='center', va='center', color='w')

        # Display the plot
        plt.title('Action-Value Array')
        plt.suptitle(
            f'Algorithm: {self.last_used_algorithm}')
        plt.show()

    def create_plot_of_episode_returns(self):
        # Set the figure size
        plt.figure(figsize=(10, 7))

        # Create a plot
        plt.plot(self.episode_returns)

        # Add labels to the plot
        plt.xlabel('Episode Number')
        plt.ylabel('Return')
        plt.title('Plot of Episode Returns')
        plt.suptitle(
            f'Algorithm: {self.last_used_algorithm}')

        legend_text = "\n".join(map(lambda p: f"{p}: {self.last_used_parameters[p] !s}", self.last_used_parameters))
        plt.legend([legend_text], loc='upper left', bbox_to_anchor=(0.7, 0.9))

        # Display the plot
        plt.show()
