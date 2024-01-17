import random
from math import isclose
import numpy as np

class Mdp:
    def __init__(self, states, actions, transition_probabilities, rewards):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.inspect_probabilities()
        self.rewards = rewards

    def possible_actions_indexes(self, state):
        return [index[0] for index, probability in np.ndenumerate(self.transition_probabilities[state]) if
                probability != 0]

    def inspect_probabilities(self):
        for state_index in range(len(self.states)):
            for action_index in range(len(self.actions)):
                probabilities_sum = sum(self.transition_probabilities[state_index, action_index, :])
                if probabilities_sum == 0:
                    continue
                else:
                    assert isclose(sum(self.transition_probabilities[state_index, action_index, :]), 1, abs_tol=1e-4)

    def step(self, current_state, action):
        probabilities = self.transition_probabilities[current_state, action]
        new_state = np.random.choice(len(probabilities), p=probabilities)
        reward = self.rewards[current_state, action, new_state]
        is_terminal = len(self.possible_actions_indexes(new_state)) == 0
        return (new_state, reward, is_terminal)