import copy
import numpy as np
from itertools import product

# INPUT FORMAT
# not finished yet

# mooc = {
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]],
#     "course name": [["skill name", "required vector", "upscale vector"], ["skill name", "required vector", "upscale vector"]]
# }

# Example MOOC Dictionary
mooc_data = {
    'name': 'Mathematics MOOC',
    'modules': {
        'Algebra': {'requirements': [2, 1, 0, 0], 'upskill_vector': [2, 1, 0, 0]},
        'Calculus': {'requirements': [1, 2, 1, 0], 'upskill_vector': [2, 1, 0, 0]},
        'Statistics': {'requirements': [0, 0, 2, 1], 'upskill_vector': [2, 1, 0, 0]},
        'Geometry': {'requirements': [1, 0, 1, 1], 'upskill_vector': [2, 1, 0, 0]},
        'Logic': {'requirements': [0, 1, 0, 2], 'upskill_vector': [2, 1, 0, 0]}
    }
}

class CourseMDPEnvironmentBuilder:
    def __init__(self, mooc, alpha, beta, gamma, learning_ability):
        # alpha parameter for updating state on succes
        self.alpha = alpha
        # beta parameter for updating state on failure
        self.beta = beta
        # learning ability x for updating states
        self.learning_ability = learning_ability

        # gamma parameter for calculating transition probability
        self.gamma = gamma

        self.mooc = mooc
        self.states = []
        self.actions = []
        self.transition_probabilities = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))

    def CreateStates(self):
        # states, actions, transition probabilities, rewards, state action array (Q-Table) set up
        max_required_vector = 0
        max_upscale_vector = 0

        # getting every skill name from the courses and max required vector and upscale vector
        skills = []
        for course in self.mooc:
            for skill in self.mooc[course]:
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
            self.states.append(state)
        # print(states)

    def CreateActions(self):
        # action is taking a course, so there is only need to get names of courses
        for course in self.mooc:
            self.actions.append(course)
        # print(actions)

    def CalculateTransitionProbability(self, source_state, action):
        # Calculate the dot product r_i · s (requirement vector of module and skill level of student)
        # for the given source_state and action
        r_i = self.mooc[action]['upskill_vector']
        dot_product = np.dot(r_i, self.GetSkillVector(source_state))

        # Calculate the transition probability using the sigmoid function
        probability = 1 / (1 + np.exp(-self.gamma * dot_product))
        return probability

    def CreateTransitionProbabilities(self):
        for i, source_state in enumerate(self.states):
            for j, action in enumerate(self.actions):
                for k, target_state in enumerate(self.states):
                    self.transition_probabilities[i, j, k] = self.CalculateTransitionProbability(source_state, action)

    def CreateRewards(self):
        # Actual reward logic still needs to be added for now manual
        self.rewards[self.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), self.actions.index("course3"), self.states.index(
            [{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])] = 1