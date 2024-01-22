import numpy as np
import matplotlib.pyplot as plt


class Solver:
    def __init__(self, mdp, episodes, max_steps_in_episode, start_state_index,
                 theta, alpha_q_learning, alpha_monte_carlo, epsilon, gamma):
        # mdp
        self.mdp = mdp

        self.action_value_array = np.zeros((len(self.mdp.states), len(self.mdp.actions)))

        # Env variables
        self.episodes = episodes
        self.max_steps_in_episode = max_steps_in_episode
        self.start_state_index = start_state_index

        # threshold for policy evaluation
        self.theta = theta
        # Learning Rate
        self.alpha_q_learning = alpha_q_learning
        self.alpha_monte_carlo = alpha_monte_carlo
        # Exploration Rate
        self.epsilon = epsilon
        # Discount Factor
        self.gamma = gamma

    def reset_solver(self, episodes, max_steps_in_episode, start_state_index,
                     theta, alpha_q_learning, alpha_monte_carlo, epsilon, gamma):
        self.action_value_array = np.zeros((len(self.mdp.states), len(self.mdp.actions)))

        # Env variables
        self.episodes = episodes
        self.max_steps_in_episode = max_steps_in_episode
        self.start_state_index = start_state_index

        # threshold for policy evaluation
        self.theta = theta
        # Learning Rate
        self.alpha_q_learning = alpha_q_learning
        self.alpha_monte_carlo = alpha_monte_carlo
        # Exploration Rate
        self.epsilon = epsilon
        # Discount Factor
        self.gamma = gamma

    def solve_with_dynamic_programming(self):
        self.__dynamic_programming()

    def solve_with_q_learning(self, episodes_enabled, policy_evaluation_enabled):
        if episodes_enabled:
            for episode in range(self.episodes):
                print(f"q_learning episode: {episode}")
                self.__q_learning()

        # policy evaluation
        episode = 0
        if policy_evaluation_enabled:
            while True:
                episode += 1
                print(f"q_learning iteration: {episode}")

                difference = self.__q_learning()
                if difference < self.theta:
                    break

    def solve_with_monte_carlo(self, episodes_enabled, policy_evaluation_enabled):
        if episodes_enabled:
            for episode in range(self.episodes):
                print(f"monte_carlo episode: {episode}")
                self.__monte_carlo()

        # policy evaluation
        episode = 0
        if policy_evaluation_enabled:
            while True:
                episode += 1
                print(f"monte_carlo iteration: {episode}")

                difference = self.__monte_carlo()
                if difference < self.theta:
                    break

    def __dynamic_programming(self):
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
                                self.mdp.rewards[state, action, new_state] + self.gamma * self.action_value_array[
                            new_state, np.argmax(self.action_value_array[new_state, :])])
                    self.action_value_array[state, action] = new_value
                    difference = max(difference, abs(old_value - new_value))
            if difference < self.theta:
                break

    def __q_learning(self):
        # start episode
        difference = 0
        previous_state = self.start_state_index
        for step in range(self.max_steps_in_episode):
            best_action = np.argmax(self.action_value_array[previous_state, :])
            possible_actions = self.mdp.possible_actions_indexes(previous_state)

            # choose action based on epsilon-greedy policy
            # np.max(action_value_array[previous_state, :]) is there to balance starting states
            # when action_value_array is full of zeros and agent always goes with action with index 0
            if np.random.rand() < self.epsilon or np.max(
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

            # update Action Value function (Q)
            new_value = (1 - self.alpha_q_learning) * self.action_value_array[
                previous_state, action] + self.alpha_q_learning * (
                                    reward + self.gamma * max(self.action_value_array[new_state, :]))
            self.action_value_array[previous_state, action] = new_value

            previous_state = new_state

            # if new state is terminal finish episode
            if is_terminal:
                break

            difference = max(difference, abs(old_value - new_value))

        return difference

    def __monte_carlo(self):
        # start episode
        monte_carlo_history = []
        previous_state = self.start_state_index
        for step in range(self.max_steps_in_episode):
            best_action = np.argmax(self.action_value_array[previous_state, :])
            possible_actions = self.mdp.possible_actions_indexes(previous_state)

            # choose action based on epsilon-greedy policy
            # np.max(action_value_array[previous_state, :]) is there to balance starting states
            # when action_value_array is full of zeros and agent always goes with action with index 0
            if np.random.rand() < self.epsilon or np.max(
                    self.action_value_array[previous_state, :]) or best_action not in possible_actions:
                # random action
                action = np.random.choice(self.mdp.possible_actions_indexes(previous_state))
            else:
                # best action
                action = best_action

            # make an action
            new_state, reward, is_terminal = self.mdp.step(previous_state, action)

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

            episode_return = self.gamma * episode_return + reward
            new_value += self.alpha_monte_carlo * (episode_return - self.action_value_array[state, action])
            self.action_value_array[state, action] = new_value

            difference = max(difference, abs(old_value - new_value))

        return difference

    def create_plot(self):
        # Set the figure size
        plt.figure(figsize=(8, 30))

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

        # Display the plot
        plt.title('Action-Value Array')
        plt.show()