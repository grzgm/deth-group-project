import copy
import numpy as np
from itertools import product

def builder():
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

    #print(states)

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

    #print(transition_probabilities)

    # rewards
    rewards = np.zeros((len(states), len(actions), len(states)))
    rewards[states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), actions.index("course3"), states.index(
        [{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])] = 1

    return copy.deepcopy(states), copy.deepcopy(actions), np.copy(transition_probabilities), np.copy(rewards)