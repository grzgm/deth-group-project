# generic stochastic mdp

# The term "stochastic" indicates that there is randomness or uncertainty in the transitions between states.
# In a generic stochastic MDP, the transition probabilities are not deterministic; they represent probabilities.

import numpy as np
from MDP_Environments_generator import MDPGridEnvironmentWithKeyGenerator


class MDP:
    def __init__(self, states, actions, transitions, rewards, slipperiness, is_slippery, cost_of_living):

        self.states = states
        self.actions = actions
        self.transitions = transitions

        # A terminal state is a state which has no actions available,
        # so we can fill the list of terminal states by checking which states have no actions
        self.terminal_states = [state for state in states if len(transitions[state]) == 0]

        # Define rewards
        self.rewards = rewards
        self.current_state = None
        self.slipperiness = slipperiness
        self.is_slippery = is_slippery
        self.cost_of_living = cost_of_living

        # Bool for keeping track of termination
        self.terminated = False

    def reset(self, initial_state=None):
        self.current_state = initial_state if initial_state is not None else np.random.choice(self.states)
        return self.current_state

    # the deterministic reward function which returns the reward for a given: state, action and next_state
    # this function uses the rewards dictionary of the mdp class
    def deterministic_reward(self, current_state, action, next_state):
        default_reward = 0  # Default reward if the combination is not present in the dictionary

        try:
            reward = self.rewards[current_state][action][next_state]
        except KeyError:
            reward = default_reward

        return reward

    def take_action(self, action):
        if self.current_state is None:
            raise ValueError("MDP not initialized. Call reset() first.")

        if action not in self.actions:
            raise ValueError("Invalid action.")

        # if it is slippery there is a chance equal to slipperiness that a different action is chosen
        if self.is_slippery:
            original_action = action
            # divide the slip prob across the actions so the total sum will be 1
            slipperiness_prob = self.slipperiness / (len(self.actions) - 1)
            action_probabilities = [1 - self.slipperiness if a == original_action else slipperiness_prob for a in
                                    self.actions]
            action = np.random.choice(self.actions, p=action_probabilities)

            if action != original_action:
                print("MDP slipped!")

        # Get the dictionary of transition probabilities for the current state and action
        transition_probs = self.transitions[self.current_state][action]

        # Extract the next states and their probabilities as separate lists
        next_states, probs = zip(*transition_probs.items())

        # randomly pick the next state using the probabilities
        next_state = np.random.choice(next_states, p=probs)

        # get the reward which is connected to moving from the current state to the chosen next state
        reward = self.deterministic_reward(self.current_state, action,
                                           next_state)  # the deterministic function is called with the current state, action and the next_state

        # take in account the cost of living
        reward -= self.cost_of_living

        # currentstate becomes the next state
        self.current_state = next_state

        # check for terminal state
        if self.current_state in self.terminal_states:
            self.terminated = True
        else:
            self.terminated = False

        # return the new state and the reward and the termination state
        return next_state, reward, self.terminated

    # function which returns the available actions in a given state
    def get_actions(self, state):
        if state not in self.states:
            raise ValueError("Invalid state.")

        return self.transitions[state]

    # function which returns the available rewards in a given state and chosen action
    def get_rewards(self, state, action):
        if state not in self.states or action not in self.actions:
            raise ValueError("Invalid state or action.")

        # Assuming reward_function is a nested dictionary
        return self.rewards[state][action]

# Example usage of the mdp class:

def testMDPwithkeyGrid():
    try:
        actions = ['left', 'right', 'up', 'down']
        grid_size = (1, 5)
        state_to_attack_key_to = 3
        locked_state = 5

        Environment = MDPGridEnvironmentWithKeyGenerator(grid_size[0], grid_size[1], state_to_attack_key_to,
                                                         locked_state, actions)
        print("Grid Environment generated Successfully!")

        is_slippery = False
        probability_of_slipping = 0.1
        cost_of_living = 0.1

        mdp = MDP(Environment.states, Environment.actions, Environment.transitions,
                  Environment.rewards, probability_of_slipping, is_slippery, cost_of_living)
        print("MDP generated Successfully!")

        # testing grabbing the key and going to the locked state
        action = 'down'
        mdp.reset('s3')
        print(f"Start state: {mdp.current_state}")
        actions_available = mdp.get_actions(mdp.current_state)
        print(f"Available actions in state {mdp.current_state}: {actions_available}")
        rewards_available = mdp.get_rewards(mdp.current_state, action)
        print(f"Rewards for state {mdp.current_state}, action {action}: {rewards_available}")

        # grab the key and go to the locked state
        newstate, recievedreward, action_done = mdp.take_action(action)
        Environment.update_environment(newstate)
        print(f"Action {action_done} taken, newstate = {newstate}")

        newstate, recievedreward, action_done = mdp.take_action('up')
        Environment.update_environment(newstate)
        print(f"Action {action_done} taken, newstate = {newstate}")

        newstate, recievedreward, action_done = mdp.take_action('right')
        Environment.update_environment(newstate)
        print(f"Action {action_done} taken, newstate = {newstate}")

        newstate, recievedreward, action_done = mdp.take_action('right')
        Environment.update_environment(newstate)
        print(f"Action {action_done} taken, newstate = {newstate}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    testMDPwithkeyGrid()
