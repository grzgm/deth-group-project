import numpy as np
from itertools import product
from mdp import Mdp

# mooc set up
# mooc = {
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]],
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]]
# }

mooc = {
    "course1": [["skillA", 0, 1], ["skillB", 0, 1]],
    "course2": [["skillA", 0, 0], ["skillC", 0, 1]],
    "course3": [["skillB", 1, 2], ["skillC", 1, 2]]
}

# states, actions, transition probabilities, rewards, state action array (Q-Table) set up
# states
states = []
max_required_vector = 0
max_upscale_vector = 0

# getting every skill name from the courses and max required vector and upscale vector
skills = []
for course in mooc:
    for skill in mooc[course]:
        if skill[0] not in skills:
            skills.append(skill[0])
        if max_required_vector < skill[1]:
            max_required_vector = skill[1]
        if max_upscale_vector < skill[2]:
            max_upscale_vector = skill[2]

# Generate all permutations of possible skill levels for amount of skills
all_permutations = product(range(max_upscale_vector + 1), repeat=len(skills))

# each permutaions of skill levels assign as a possible state
for skill_levels in all_permutations:
    state = []
    for i in range(len(skill_levels)):
        state.append({skills[i]: skill_levels[i]})
    states.append(state)

# print(states)

# actions
actions = []

# action is taking a course, so there is only need to get names of courses
for course in mooc:
    actions.append(course)

# print(actions)

# transition probabilities
transition_probabilities = np.zeros((len(states), len(actions), len(states)))

transition_probabilities[
    states.index([{'skillA': 0}, {'skillB': 0}, {'skillC': 0}]), actions.index("course1"), states.index(
        [{'skillA': 1}, {'skillB': 1}, {'skillC': 0}])] = 1
transition_probabilities[
    states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 0}]), actions.index("course2"), states.index(
        [{'skillA': 1}, {'skillB': 1}, {'skillC': 1}])] = 1
transition_probabilities[
    states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), actions.index("course3"), states.index(
        [{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])] = 1

# transition_probabilities = np.zeros((len(states), len(actions), len(states)))
#
# for state_index in range(len(states)):
#     for action_index in range(len(actions)):
#         for next_state_index in range(len(states)):
#             if state_index in [0, 6]:
#                 continue
#             elif action_index == 1 and state_index == next_state_index - 1:
#                 transition_probabilities[state_index, action_index, next_state_index] = 1
#             elif action_index == 0 and state_index == next_state_index + 1:
#                 transition_probabilities[state_index, action_index, next_state_index] = 1
#

# rewards
rewards = np.zeros((len(states), len(actions), len(states)))
rewards[states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), actions.index("course3"), states.index(
    [{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])] = 1

state_value_array = []
action_value_array = np.zeros((len(states), len(actions)))

# Env variables
episodes = 1000
max_steps_in_episode = 1000
start_state_index = 0

q_learning_enabled = True
monte_carlo_enabled = True

# Learning Rate
alpha_q_learning = 0.04
alpha_monte_carlo = 0.1
# Exploration Rate
epsilon = 0.00
# Discount Factor
gamma = 0.9

mdp = Mdp(states, actions, transition_probabilities, rewards, random_termination=0.3, cost_of_living=-1.5)

for episode in range(episodes):
    print(f"episode: {episode}")
    previous_state = start_state_index
    total_reward = 0
    monte_carlo_history = []

    # start episode
    for step in range(max_steps_in_episode):
        best_action = np.argmax(action_value_array[previous_state, :])
        possible_actions = mdp.possible_actions(previous_state)

        # choose action based on epsilon-greedy policy
        # np.max(action_value_array[previous_state, :]) is there to balance starting states
        # when action_value_array is full of zeros and agent always goes with action with index 0
        if np.random.rand() < epsilon or np.max(action_value_array[previous_state, :]) or best_action not in possible_actions:
            # random action
            action = np.random.choice(mdp.possible_actions(previous_state))
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
