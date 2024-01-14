import numpy as np
from mdp import Mdp


# if __name__ == "__main__":
# states, actions, transition probabilities, rewards, state action array (Q-Table) set up
states = ['0', '1', '2', '3', '4', '5', '6']
actions = ['d', 'u']

transition_probabilities = np.zeros((len(states), len(actions), len(states)))

for state_index in range(len(states)):
    for action_index in range(len(actions)):
        for next_state_index in range(len(states)):
            if state_index in [0, 6]:
                continue
            elif action_index == 1 and state_index == next_state_index - 1:
                transition_probabilities[state_index, action_index, next_state_index] = 1
            elif action_index == 0 and state_index == next_state_index + 1:
                transition_probabilities[state_index, action_index, next_state_index] = 1

rewards = np.zeros((len(states), len(actions), len(states)))
rewards[states.index('1'), actions.index('d'), states.index('0')] = 1
rewards[states.index('5'), actions.index('u'), states.index('6')] = -1

state_value_array = []
action_value_array = np.zeros((len(states), len(actions)))

# Env variables
episodes = 1000
max_steps_in_episode = 1000
start_state_index = 3

q_learning_enabled = True
monte_carlo_enabled = True

# Learning Rate
alpha_q_learning = 0.04
alpha_monte_carlo = 0.1
# Exploration Rate
epsilon = 0.05
# Discount Factor
gamma = 0.9

mdp = Mdp(states, actions, transition_probabilities, rewards, random_termination=0.3, cost_of_living=-1.5)

for episode in range(episodes):
    print(episode)
    previous_state = start_state_index
    total_reward = 0
    monte_carlo_history = []

    # start episode
    for step in range(max_steps_in_episode):
        # choose action based on epsilon-greedy policy
        # np.max(action_value_array[previous_state, :]) is there to balance starting states
        # when action_value_array is full of zeros and agent always goes with action with index 0
        if np.random.rand() < epsilon or np.max(action_value_array[previous_state, :]):
            # random action
            action = np.random.choice(mdp.possible_actions_indexes(previous_state))
        else:
            # best action
            action = np.argmax(action_value_array[previous_state, :])

        # make an action
        new_state, reward, is_terminal = mdp.step(previous_state, action)

        # update Action Value function (Q) for Q-learning
        if q_learning_enabled:
            action_value_array[previous_state, action] = (1 - alpha_q_learning) * action_value_array[
                previous_state, action] + alpha_q_learning * (reward + gamma * max(action_value_array[new_state, :]))

        # add the episode states, actions, rewards for  Monte Carlo
        if monte_carlo_enabled:
            monte_carlo_history.append((previous_state, action, reward))

        previous_state = new_state

        # if new state is terminal finish episode
        if (is_terminal):
            break

    if monte_carlo_enabled:
        # Calculate returns and update Action Value function (Q)
        episode_return = 0
        for t in range(len(monte_carlo_history) - 1, -1, -1):
            state = monte_carlo_history[t][0]
            action = monte_carlo_history[t][1]
            reward = monte_carlo_history[t][2]

            episode_return = gamma * episode_return + reward
            action_value_array[state, action] += alpha_monte_carlo * (
                        episode_return - action_value_array[state, action])

# Action Value Array (Q-Table) at the end
for a in actions:
    print(f'\t{a}\t', end='')
print()
for s in states:
    print(s, end=' ')
    with np.printoptions(precision=3, suppress=True):
        print(action_value_array[states.index(s)])
