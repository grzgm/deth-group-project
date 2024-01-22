import copy
import numpy as np
import math
from itertools import product

class EnvironmentBuilder:
    def __init__(self, mooc, transition_probability_gamma, alpha, beta, student_learning_ability):
        # Hyperparameters
        self.alpha = alpha  # for state updating when passing
        self.beta = beta  # for state updating when failing
        self.student_learning_ability = student_learning_ability  # student learning ability for state updating
        self.gamma = transition_probability_gamma  # gamma parameter for calculating transition probability

        # extract data to construct the states, actions, transition_probabilities, rewards
        self.mooc = mooc
        self.skills = []
        self.max_skill_level = 0
        self.extarct_skills_and_max_skill_level()

        self.states = []
        self.actions = []
        self.transition_probabilities = None
        self.rewards = None


    # get the requirements for a given action (course)
    def get_requirement_vector(self, action):
        # Get the requirements for the given action
        action_requirements = [next((skill[1] for skill in self.mooc[action] if skill[0] == s), 0) for s in self.skills]

        return np.array(action_requirements)

    def get_student_skill(self, state):
        # Extract skill values from the state
        student_skill_levels = np.array([next(iter(skill.values())) for skill in state])
        return student_skill_levels

    def get_upskill_vector(self, action):
        # Get the upskill vector for the given action (course)
        upskill_vector = [next((skill[2] for skill in self.mooc[action] if skill[0] == s), 0) for s in self.skills]

        return upskill_vector

    def get_next_state(self, source_state, action, passed):
        # Get the upskill vector for the given action (course)
        upskill_vector = self.get_upskill_vector(action)

        # Update the state based on the action result (passing)
        new_state = copy.deepcopy(source_state)
        for idx, skill_level in enumerate(new_state):
            skill_name = list(skill_level.keys())[0]
            current_skill_level = skill_level[skill_name]

            # Check if the skill level has reached the maximum
            if current_skill_level < self.max_skill_level:
                # if the max has not been reached update the skill level based on passing or failing
                update_value = self.alpha * upskill_vector[idx] * self.student_learning_ability if passed \
                    else self.beta * upskill_vector[idx] * self.student_learning_ability
                new_state[idx][skill_name] += update_value

                # Ensure the skill level is an integer
                new_state[idx][skill_name] = math.floor(new_state[idx][skill_name])

                # if the new state exceeds the maximum skill level, set it to the maximum
                if new_state[idx][skill_name] > self.max_skill_level:
                    new_state[idx][skill_name] = self.max_skill_level

        return new_state

    def extarct_skills_and_max_skill_level(self):
        # getting every skill name from the courses and max required vector and upscale vector
        for course in self.mooc:
            for skill in self.mooc[course]:
                if skill[0] not in self.skills:
                    self.skills.append(skill[0])
                self.max_skill_level = max(self.max_skill_level, skill[1], skill[2])

    def create_states(self):
        # Generate all permutations of possible skill levels for amount of skills
        all_permutations = product(range(self.max_skill_level + 3), repeat=len(self.skills))

        # each permutaions of skill levels assign as a possible state
        for skill_levels in all_permutations:
            state = []
            for i in range(len(skill_levels)):
                state.append({self.skills[i]: skill_levels[i]})
            self.states.append(state)
        # print(self.states)

    def create_actions(self):
        # action is taking a course, so there is only need to get names of courses
        for course in self.mooc:
            self.actions.append(course)
        # print(self.actions)

    def calculate_transition_probability(self, source_state, action):
        # Retrieve the 'required vector' from the MOOC dictionary for the specific action
        required_vector = self.get_requirement_vector(action)

        # Get the student's skill levels from the source_state
        student_skill_levels = self.get_student_skill(source_state)

        # Calculate the dot product of the requirement vector and the skill level vector
        dot_product = np.dot(required_vector, student_skill_levels)

        # Calculate the probability of passing using the sigmoid function
        probability_of_passing = 1 / (1 + np.exp(-self.gamma * dot_product))

        # Probability of failing is complementary to the probability of passing
        probability_of_failing = 1 - probability_of_passing

        return probability_of_passing, probability_of_failing

    def create_transition_probabilities(self):
        # create the transition probabilities
        self.transition_probabilities = np.zeros((len(self.states), len(self.actions), len(self.states)))

        for state in self.states:
            for action in self.actions:
                # Get the transition probabilities for the given state and action
                probability_of_passing, probability_of_failing = self.calculate_transition_probability(state, action)

                # Get the next state for the given action when passing and failing
                next_state_pass = self.get_next_state(state, action, True)
                next_state_fail = self.get_next_state(state, action, False)

                # Get the index of the current state
                current_state_index = self.states.index(state)

                # Get the index of the next states
                next_state_index_pass = self.states.index(next_state_pass)
                next_state_index_fail = self.states.index(next_state_fail)

                # Get the index of the current action
                current_action_index = self.actions.index(action)

                if next_state_index_pass == current_state_index and next_state_index_fail == current_state_index:
                    continue

                # Set the transition probability for the given state, action, and next state (passing)
                self.transition_probabilities[
                    current_state_index, current_action_index, next_state_index_pass] = probability_of_passing

                # Set the transition probability for the given state, action, and current state (failing)
                self.transition_probabilities[
                    current_state_index, current_action_index, next_state_index_fail] = probability_of_failing

        # Print the transition probabilities
        # print(self.transition_probabilities)

    def manual_rewards(self):
        # Actual reward logic still needs to be added for now manual
        self.rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))

        self.rewards[self.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), self.actions.index("course3"), self.states.index(
            [{'skillA': 1}, {'skillB': 3}, {'skillC': 3}])] = 1

    # This version of the create rewards correctly assigns the rewards, but seems to give some problems to the demo when q-learning
    def create_rewards(self):
        # Initialize the rewards array
        self.rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))

        # Iterate over all states and actions
        for s_idx, source_state in enumerate(self.states):
            for a_idx, action in enumerate(self.actions):
                for ns_idx, next_state in enumerate(self.states):
                    # Check if the next state is a passing state based on the action
                    passed = self.get_next_state(source_state, action, True) == next_state

                    # Assign reward based on passing or failing, 3 for passing and 1 for failing
                    self.rewards[s_idx, a_idx, ns_idx] = 3 if passed else 1

    def get_everything(self):
        self.create_states()
        self.create_actions()
        self.create_transition_probabilities()
        # self.manual_rewards()
        self.create_rewards()
        return self.states, self.actions, self.transition_probabilities, self.rewards


# testing the builder functions
if __name__ == '__main__':
    mooc = {
            "course1": [["skillA", 0, 1], ["skillB", 0, 1]],
            "course2": [["skillA", 0, 0], ["skillC", 0, 1]],
            "course3": [["skillB", 1, 2], ["skillC", 1, 2]]
        }

    # hyperparameters
    alpha = 2.0
    beta = 1.1
    student_learning_ability = 1

    builder = EnvironmentBuilder(mooc, 0.1, alpha, beta, student_learning_ability)
    builder.create_states()
    builder.create_actions()
    builder.create_transition_probabilities()
    builder.manual_rewards()
    print(builder.get_next_state([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}], "course3", True))

    print("Probability of passing course3 with skill levels [1, 1, 1]:")
    print(builder.transition_probabilities[builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
    builder.states.index([{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])])

    print("Probability of failing course3 with skill levels [1, 1, 1]:")
    print(builder.transition_probabilities[builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
    builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}])])

    print("Reward of passing course3 with skill levels [1, 1, 1]:")
    print(builder.rewards[
              builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
              builder.states.index([{'skillA': 1}, {'skillB': 2}, {'skillC': 2}])])

    print("Reward of Failing course3 with skill levels [1, 1, 1]:")
    print(builder.rewards[
              builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}]), builder.actions.index("course3"),
              builder.states.index([{'skillA': 1}, {'skillB': 1}, {'skillC': 1}])])

    for s in range(len(builder.states)):
        for a in range(len(builder.actions)):
            #print(np.sum(builder.transition_probabilities[s, a, :]))
            for ns in range(len(builder.states)):
                pass
                #if builder.transition_probabilities[s, a, ns] != 0:
                   # print(builder.transition_probabilities[s, a, ns])
