### Monte-Carlo

During the episode of the agent history of choices is recorded, and at the end based on them the **action value array** is updated, by analysing the agent path from the end to the beginning. This allows for easy computation of the reward sum on the go. In addition, it does not interfere with the Q-Learning. It is controlled by the variable `monte_carlo_enabled`, which allows for control over agent. Monte-Carlo formula:
$$
action_value_array[state, action] += alpha_monte_carlo * (
                    episode_return - action_value_array[state, action])
$$

### Q-Learning

It is controlled by the variable `q_learning_enabled` and if enabled updates the **action value array** every step of the agent. Q-Learning formula:
$$
action_value_array[previous_state, action] = (1 - alpha_q_learning) * action_value_array[
                previous_state, action] + alpha_q_learning * (reward + gamma * max(action_value_array[new_state, :]))
$$

### Generalized Policy Iteration

#### Policy Improvement

GPI is implemented by starting with the Policy Improvement 
(in the beginning whole **action value array** is full of zeros, which means any action is as good as any other)
that utilises the `argmax()` function to achieve the best policy possible in the current moment, which is greedy policy.

#### Policy Evaluation

After Policy Improvement policy is used to calculate new values of the **action value array**, which is updated in the process and functions as a Policy Evaluation.

#### GPI Scheme

##### Q-Learning

For the Q-Learning, the policy undergoes Policy Improvement and Policy Evaluation every step of the agent, as every step of the agent the **action value array** is updated, and based on the new values the `argmax()` function selects new optimal policy, creating cycle of the Policy Improvement and Policy Evaluation.

##### Monte-Carlo

The case of Monte-Carlo is similar, but the Policy Improvement and Policy Evaluation take place after each episode.

### Conclusion
The model is incorporating the test data, and the agent is learning the optimal path to achieve the goal, but the model need further testing on more diverse scenarios. The ground basics for the project has been laid out.