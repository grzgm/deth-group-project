from mdp import Mdp
from environment_builder import EnvironmentBuilder
from solver import Solver

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
alpha_state_update = 5
beta_state_update = 0.1
student_learning_ability = 1

# states, actions, transition_probabilities, rewards = builder()
builder = EnvironmentBuilder(mooc, transition_probability_gamma, alpha_state_update, beta_state_update,
                             student_learning_ability)
states, actions, transition_probabilities, rewards = builder.get_everything()

mdp = Mdp(states, actions, transition_probabilities, rewards)

solver = Solver(mdp)

# dynamic programming
solver.solve_with_dynamic_programming(theta=0.000001, gamma=0.9)
solver.create_plot()

# q-learning
solver.reset_solver()
solver.solve_with_q_learning(True, True,
                             episodes=10,
                             theta=0.0052,
                             max_steps_in_episode=1000,
                             start_state_index=0,
                             alpha=0.04,
                             epsilon=0.00,
                             gamma=0.9)
solver.create_plot()

solver.reset_solver()
solver.solve_with_q_learning(True, True,
                             episodes=10,
                             theta=0.005,
                             max_steps_in_episode=1000,
                             start_state_index=0,
                             alpha=0.04,
                             epsilon=0.00,
                             gamma=0.9)
solver.create_plot()

# monte carlo
solver.reset_solver()
solver.solve_with_monte_carlo(True, True,
                              episodes=10,
                              theta=0.015,
                              max_steps_in_episode=1000,
                              start_state_index=0,
                              alpha=0.04,
                              epsilon=0.00,
                              gamma=0.9)
solver.create_plot()

# # Action Value Array at the end
# for a in actions:
#     print(f'\t{a}\t', end='')
# print()
# for s in states:
#     print(s, end=' ')
#     with np.printoptions(precision=3, suppress=True):
#         print(action_value_array[states.index(s)])
