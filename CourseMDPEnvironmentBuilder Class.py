import copy
import numpy as np
from itertools import product

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

        self.skills = []
        self.mooc = mooc
        self.states = []
        self.actions = []
        self.transition_probabilities = None
        self.rewards = None

        self.max_skill_levels = [4, 4, 4]

    # get the requirements for a given action (course)
    def GetRequirements(self, action):
        # Get the requirements for the given action
        action_requirements = [next((skill[1] for skill in self.mooc[action] if skill[0] == s), 0) for s in self.skills]

        return np.array(action_requirements)

    def GetStudentSkill(self, state):
        # Extract skill values from the state
        student_skill_levels = np.array([next(iter(skill.values())) for skill in state])
        return student_skill_levels

    def GetUpskillVector(self, action):
        # Get the upskill vector for the given action (course)
        upskill_vector = [next((skill[2] for skill in self.mooc[action] if skill[0] == s), 0) for s in self.skills]

        return upskill_vector

    def GetNextState(self, source_state, action):
        # Get the upskill vector for the given action (course)
        upskill_vector = self.GetUpskillVector(action)

        # Update the state based on the action result (passing)
        new_state = copy.deepcopy(source_state)
        for idx, skill_level in enumerate(new_state):
            skill_name = list(skill_level.keys())[0]
            current_skill_level = skill_level[skill_name]

            # Check if the skill level has reached the maximum
            if current_skill_level < self.max_skill_levels[idx]:
                new_state[idx][skill_name] += min(upskill_vector[idx], self.max_skill_levels[idx] - current_skill_level)

        return new_state

    def CreateStates(self):
        # states, actions, transition probabilities, rewards, state action array (Q-Table) set up
        max_required_vector = 0
        max_upscale_vector = 0

        # getting every skill name from the courses and max required vector and upscale vector
        for course in self.mooc:
            for skill in self.mooc[course]:
                if skill[0] not in self.skills:
                    self.skills.append(skill[0])
                if max_required_vector < skill[1]:
                    max_required_vector = skill[1]
                if max_upscale_vector < skill[2]:
                    max_upscale_vector = skill[2]

        # Generate all permutations of possible skill levels for amount of skills
        all_permutations = product(range(max_upscale_vector + 3), repeat=len(self.skills))

        # each permutaions of skill levels assign as a possible state
        for skill_levels in all_permutations:
            state = []
            for i in range(len(skill_levels)):
                state.append({self.skills[i]: skill_levels[i]})
            self.states.append(state)
        print(self.states)

    def CreateActions(self):
        # action is taking a course, so there is only need to get names of courses
        for course in self.mooc:
            self.actions.append(course)
        # print(actions)

    def CalculateTransitionProbability(self, source_state, action):
        # Retrieve the 'required vector' from the MOOC dictionary for the specific action
        required_vector = self.GetRequirements(action)

        # Get the student's skill levels from the source_state
        student_skill_levels = self.GetStudentSkill(source_state)

        # Calculate the dot product of the requirement vector and the skill level vector
        dot_product = np.dot(required_vector, student_skill_levels)

        # Calculate the probability of passing using the sigmoid function
        probability_of_passing = 1 / (1 + np.exp(-self.gamma * dot_product))

        # Probability of failing is complementary to the probability of passing
        probability_of_failing = 1 - probability_of_passing

        return probability_of_passing, probability_of_failing

    def CreateTransitionProbabilities(self):
        # create the transition probabilities
        self.transition_probabilities = np.zeros((len(self.states), len(self.actions), len(self.states)))

        for state in self.states:
            for action in self.actions:
                # Get the transition probabilities for the given state and action
                probability_of_passing, _ = self.CalculateTransitionProbability(state, action)

                # Get the next state for the given action
                next_state = self.GetNextState(state, action)

                # Get the index of the current state
                current_state_index = self.states.index(state)

                # Get the index of the next state
                next_state_index = self.states.index(next_state)

                # Get the index of the current action
                current_action_index = self.actions.index(action)

                # Set the transition probability for the given state, action, and next state (passing)
                self.transition_probabilities[
                    current_state_index, current_action_index, next_state_index] = probability_of_passing

                # Set the transition probability for the given state, action, and current state (failing)
                self.transition_probabilities[
                    current_state_index, current_action_index, current_state_index] = 1 - probability_of_passing

        # Print the transition probabilities
        print(self.transition_probabilities)

    def CreateRewards(self):
        # Actual reward logic still needs to be added for now manual
        self.rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))

        self.rewards[self.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), self.actions.index("course3"), self.states.index(
            [{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])] = 1

# testing the builder functions

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

builder = CourseMDPEnvironmentBuilder(mooc, 0.1, 0.1, 0.1, 0.1)
builder.CreateStates()
builder.CreateActions()
builder.CreateTransitionProbabilities()
#print(builder.GetNextState([{'skillA': 4}, {'skillB': 4}, {'skillC': 4}], "course3"))

print("Probability of passing course3 with skill levels [1, 1, 1]:")
print(builder.transition_probabilities[builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
builder.states.index([{'skillA': 1}, {'skillB': 3}, {'skillC': 3}])])

print("Probability of failing course3 with skill levels [1, 1, 1]:")
print(builder.transition_probabilities[builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}])])
