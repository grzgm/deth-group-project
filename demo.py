import numpy as np
from mdp import Mdp
from environment_builder import EnvironmentBuilder
import matplotlib.pyplot as plt


def dynamic_programming():
    # policy evaluation
    episode = 0
    while True:
        print(f"episode: {episode}")
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
                difference = max(difference, abs(old_value - new_value))
        if difference < theta:
            break


def q_learning():
    # start episode
    difference = 0
    previous_state = start_state_index
    for step in range(max_steps_in_episode):
        best_action = np.argmax(action_value_array[previous_state, :])
        possible_actions = mdp.possible_actions_indexes(previous_state)

        # choose action based on epsilon-greedy policy
        # np.max(action_value_array[previous_state, :]) is there to balance starting states
        # when action_value_array is full of zeros and agent always goes with action with index 0
        if np.random.rand() < epsilon or np.max(
                action_value_array[previous_state, :]) or best_action not in possible_actions:
            # random action
            action = np.random.choice(mdp.possible_actions_indexes(previous_state))
        else:
            # best action
            action = best_action

        old_value = action_value_array[previous_state, action]
        new_value = 0

        # make an action
        new_state, reward, is_terminal = mdp.step(previous_state, action)

        # update Action Value function (Q)
        new_value = (1 - alpha_q_learning) * action_value_array[
            previous_state, action] + alpha_q_learning * (reward + gamma * max(action_value_array[new_state, :]))
        action_value_array[previous_state, action] = new_value

        previous_state = new_state

        # if new state is terminal finish episode
        if (is_terminal):
            break

        difference = max(difference, abs(old_value - new_value))

    return difference


def monte_carlo():
    # start episode
    monte_carlo_history = []
    previous_state = start_state_index
    for step in range(max_steps_in_episode):
        best_action = np.argmax(action_value_array[previous_state, :])
        possible_actions = mdp.possible_actions_indexes(previous_state)

        # choose action based on epsilon-greedy policy
        # np.max(action_value_array[previous_state, :]) is there to balance starting states
        # when action_value_array is full of zeros and agent always goes with action with index 0
        if np.random.rand() < epsilon or np.max(
                action_value_array[previous_state, :]) or best_action not in possible_actions:
            # random action
            action = np.random.choice(mdp.possible_actions_indexes(previous_state))
        else:
            # best action
            action = best_action

        # make an action
        new_state, reward, is_terminal = mdp.step(previous_state, action)

        # add the episode states, actions, rewards for  Monte Carlo
        monte_carlo_history.append((previous_state, action, reward))

        previous_state = new_state

        # if new state is terminal finish episode
        if (is_terminal):
            break

    # Calculate returns and update Action Value function (Q)
    difference = 0
    episode_return = 0
    for t in range(len(monte_carlo_history) - 1, -1, -1):
        state = monte_carlo_history[t][0]
        action = monte_carlo_history[t][1]
        reward = monte_carlo_history[t][2]

        old_value = action_value_array[state, action]
        new_value = action_value_array[state, action]

        episode_return = gamma * episode_return + reward
        new_value += alpha_monte_carlo * (episode_return - action_value_array[state, action])
        action_value_array[state, action] = new_value

        difference = max(difference, abs(old_value - new_value))

    return difference

# mooc = {
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]],
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]]
# }

# Example MOOC Dictionary
mooc = {
        "course1": [["skillA", 0, 1], ["skillB", 0, 1]],
        "course2": [["skillA", 0, 0], ["skillC", 0, 1]],
        "course3": [["skillB", 1, 2], ["skillC", 1, 2]]
    }

# hyperparameters environment builder
transition_probability_gamma = 0.1
alpha_state_update = 0.5
beta_state_update = 0.1
student_learning_ability = 1

# states, actions, transition_probabilities, rewards = builder()
builder = EnvironmentBuilder(mooc, transition_probability_gamma, alpha_state_update, beta_state_update,
                             student_learning_ability)
states, actions, transition_probabilities, rewards = builder.get_everything()

state_value_array = []
action_value_array = np.zeros((len(states), len(actions)))

# Env variables
episodes = 10
max_steps_in_episode = 1000
start_state_index = 0

dynamic_programming_enabled = False
q_learning_enabled = False
monte_carlo_enabled = False

# threshold for policy evaluation
theta = 0.0000001
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
    dynamic_programming()

# q-learning
episode = 0
if q_learning_enabled:
    # policy evaluation
    for episode in range(episodes):
        print(f"episode: {episode}")
        q_learning()

    # policy evaluation
    while True:
        episode += 1
        print(f"iteration: {episode}")

        difference = q_learning()
        if difference < theta:
            break

# monte carlo
episode = 0
if monte_carlo_enabled:
    # policy evaluation
    for episode in range(episodes):
        print(f"episode: {episode}")
        monte_carlo()

    # policy evaluation
    while True:
        episode += 1
        print(f"iteration: {episode}")

        difference = monte_carlo()
        if difference < theta:
            break

# Action Value Array at the end
for a in actions:
    print(f'\t{a}\t', end='')
print()
for s in states:
    print(s, end=' ')
    with np.printoptions(precision=3, suppress=True):
        print(action_value_array[states.index(s)])

# Set the figure size
plt.figure(figsize=(8, 30))

# Create a heatmap
plt.imshow(action_value_array, cmap='viridis', interpolation='nearest', aspect=0.2)

# Add colorbar
plt.colorbar()

# Set axis labels
plt.xlabel('Actions')
plt.ylabel('States')

# Set custom labels for the x-axis
plt.xticks(np.arange(len(actions)), actions)

# Set custom labels for the y-axis
plt.yticks(np.arange(len(states)), states)

# Display the plot
plt.title('Action-Value Array')
plt.show()