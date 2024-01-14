import random
from math import isclose
import numpy as np

class Mdp:
    def __init__(self, states, actions, transition_probabilities, rewards, gamma=0.9,
                 eps=1e6, random_termination=0.0, cost_of_living=0.0):
        self.states = states
        self.actions = actions
        self.transition_probabilities = transition_probabilities
        self.inspect_probabilities()
        self.rewards = rewards

        self.gamma = gamma
        self.eps = eps
        self.random_termination = random_termination
        assert 0 <= self.random_termination <= 1
        self.cost_of_living = cost_of_living

        # self.value = {}
        # for state in states:
        #     self.value[state] = 0.0

    # def reset(self):
    #     for state in self.states:
    #         self.value[state] = 0.0

    def lookup_transition_probability(self, state, action, next_state):
        return self.transition_probabilities.get(state, {}).get(action, {}).get(next_state, 0.0)

    def possible_actions_indexes(self, state):
        return [index[0] for index, probability in np.ndenumerate(self.transition_probabilities[state]) if
                probability != 0]

    def lookup_reward(self, state, action, next_state):
        return self.rewards[state, action, next_state]

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
        reward = self.lookup_reward(current_state, action, new_state)
        is_terminal = len(self.possible_actions_indexes(new_state)) == 0
        return (new_state, reward, is_terminal)

    # def value(self, state):
    #     pass

    # def action_value(self, state, action):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)

    # def state_value(self, state, action):
    #     next_states = self.transition_probabilities[state].get(action, {})
    #     return sum(self.lookup_transition_probability(state, action, next_state) * (
    #             self.lookup_reward(state, action, next_state) + self.gamma * self.value[next_state]) for next_state
    #                in next_states)

    # return random.choice(self.actions)

    # def estimate_value(self):
    #     for _ in range(int(self.eps)):
    #         for state in self.states:
    #             self.value[state] = self.action_value(state, self.policy(state))