import numpy as np
from mdp import Mdp
from builder import builder

states, actions, transition_probabilities, rewards = builder()

state_value_array = []
action_value_array = np.zeros((len(states), len(actions)))

# Env variables
episodes = 1000
max_steps_in_episode = 1000
start_state_index = 0

dynamic_programming_enabled = True
q_learning_enabled = False
monte_carlo_enabled = False

# threshold for policy evaluation
theta = 0.01
# Learning Rate
alpha_q_learning = 0.04
alpha_monte_carlo = 0.1
# Exploration Rate
epsilon = 0.00
# Discount Factor
gamma = 0.9

mdp = Mdp(states, actions, transition_probabilities, rewards)

# dynamic programming
if dynamic_programming_enabled:
    # policy evaluation
    while True:
        difference = 0
        for state in range(len(states)):
            for action in range(len(actions)):
                old_value = action_value_array[state, action]
                new_value = 0
                for new_state in range(len(states)):
                    new_value += transition_probabilities[state, action, new_state] * (
                            rewards[state, action, new_state] + gamma * action_value_array[
                        new_state, np.argmax(action_value_array[new_state, :])])
                action_value_array[state, action] = new_value
                difference = max(difference, abs(old_value-new_value))
        if difference < theta:
            break

for episode in range(episodes):
    print(f"episode: {episode}")
    previous_state = start_state_index
    total_reward = 0
    monte_carlo_history = []

    # start episode
    for step in range(max_steps_in_episode):
        best_action = np.argmax(action_value_array[previous_state, :])
        possible_actions = mdp.possible_actions_indexes(previous_state)

        # choose action based on epsilon-greedy policy
        # np.max(action_value_array[previous_state, :]) is there to balance starting states
        # when action_value_array is full of zeros and agent always goes with action with index 0
        if np.random.rand() < epsilon or np.max(action_value_array[previous_state, :]) or best_action not in possible_actions:
            # random action
            action = np.random.choice(mdp.possible_actions_indexes(previous_state))
        else:
            # best action
            action = best_action

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
