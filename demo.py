from mdp import Mdp
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

# The IT MOOC Dictionary
mooc = {
    "Python Programming Basics": [
        ["Python Programming Fundamentals", 0, 2]
    ],
    "Database Management Essentials": [
        ["Python Programming Fundamentals", 2, 0],
        ["Database Management", 0, 3]
    ],
    "Introduction to Machine Learning with Python": [
        ["Python Programming Fundamentals", 2, 3],
        ["Machine Learning", 0, 4]
    ],
    "Web Development Using Python": [
        ["Python Programming Fundamentals", 2, 3],
        ["Web Development", 0, 3]
    ],
    "Advanced Python Programming and Optimization": [
        ["Python Programming Fundamentals", 3, 5]
    ],
    "Database Administration and Performance Tuning": [
        ["Database Management", 3, 6],
        ["Python Programming Fundamentals", 3, 0]
    ],
    "Deep Learning and Neural Networks with Python": [
        ["Python Programming Fundamentals", 5, 6],
        ["Machine Learning", 3, 6]
    ],
    "Full-Stack Web Development Mastery": [
        ["Python Programming Fundamentals", 4, 6],
        ["Web Development", 4, 6]
    ]
}


# hyperparameters environment builder
transition_probability_gamma = 0.1
alpha_state_update = 5
beta_state_update = 0.1
student_learning_ability = 1

# states, actions, transition_probabilities, rewards = builder()
mdp = Mdp(mooc, transition_probability_gamma, alpha_state_update, beta_state_update,
          student_learning_ability)

solver = Solver(mdp)

# dynamic programming
solver.solve_with_dynamic_programming(theta=0.000001, gamma=0.9)
solver.create_plot_of_action_value_array()

# q-learning
solver.reset_solver()
solver.solve_with_q_learning(True, True,
                             episodes=10,
                             theta=0.052,
                             max_steps_in_episode=1000,
                             start_state_index=0,
                             alpha=0.04,
                             epsilon=0.00,
                             gamma=0.9,
                             cost_of_living=-1.5
                             )
solver.create_plot_of_action_value_array()
solver.create_plot_of_episode_returns()

solver.reset_solver()
solver.solve_with_q_learning(True, True,
                             episodes=10,
                             theta=0.005,
                             max_steps_in_episode=1000,
                             start_state_index=0,
                             alpha=0.04,
                             epsilon=0.00,
                             gamma=0.9,
                             cost_of_living=0
                             )
solver.create_plot_of_action_value_array()
solver.create_plot_of_episode_returns()

# monte carlo
solver.reset_solver()
solver.solve_with_monte_carlo(True, True,
                              episodes=10,
                              theta=0.015,
                              max_steps_in_episode=1000,
                              start_state_index=0,
                              alpha=0.04,
                              epsilon=0.00,
                              gamma=0.9,
                              cost_of_living=0
                              )
solver.create_plot_of_action_value_array()
solver.create_plot_of_episode_returns()
