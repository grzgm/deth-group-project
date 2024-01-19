# Group Project

Grzegorz Malisz & Tiemon Steeghs

[GitHub Repository](https://github.com/grzgm/deth-group-project)

## Context

We are focusing on creating a model that would ease and enhance the process of learning from MOOCs [Massive Open Online Courses]. The main goal of the model is to find the best learning path for user, based on user's current profile and his goal of finishing certain course. The path established by the agent should minimize the dropout possibility, learning curve and maximise the speed of learning. For our project we have picked the domain of IT MOOCs, but our solution can be extended to other topics.

## IT MOOC

For the development purpose we have used this structure of MOOC:

1. Statistical Methods in Number Theory for Beginners
   - Arithmetic
   - Statistics
2. Algebraic Statistics
   - Algebra
   - Statistics
3. Basic Arithmetic
   - Arithmetic
   - Geometry
   - Algebra
   - Calculus
4. Elementary Algebra
   - Arithmetic
   - Geometry
   - Algebra
5. Geometry:
   - Arithmetic
   - Geometry

## MDP Formal Definition

### States

Skill levels in various domains: $(s_1, s_2, \ldots, s_N)$

Defined math skills:

1. Arithmetic
2. Geometry
3. Algebra
4. Calculus

Descriptive visualisation of the state: $(Arithmetic, Geometry, Algebra, Calculus, Statistics, Discrete Mathematics, Logic, Mathematical Analysis)$

Later in the paper we will use the two notation, which ease understanding and writing while serving different purposes. For example, to refer to the Arithmetic skill level we will use $s_1$, as well as $s_{Arithmetic}$ notation.

### Actions

Actions are defined by taking specific modules, each module has an associated upskilling vector $u_i$ and requirement vector $r_i$. Note that if value of the certain skill in the vector is 0, it is not denoted for the sake of clarity.

1. Statistical Methods in Number Theory for Beginners
   - Requirements vector:
     - $r_{Arithmetic} = 1$
     - $r_{Statistics} = 1$
   - Upskill vector:
     - $u_{Arithmetic} = 2$
     - $u_{Statistics} = 2$
2. Algebraic Statistics
   - Requirements vector:
     - $r_{Algebra} = 2$
     - $r_{Statistics} = 3$
   - Upskill vector:
     - $u_{Algebra} = 4$
     - $u_{Statistics} = 4$
3. Basic Arithmetic
   - Requirements vector:
     - None
   - Upskill vector:
     - $u_{Arithmetic} = 3$
     - $u_{Geometry} = 1$
     - $u_{Algebra} = 1$
     - $u_{Calculus} = 1$
4. Elementary Algebra
   - Requirements vector:
     - None
   - Upskill vector:
     - $u_{Arithmetic} = 1$
     - $u_{Geometry} = 1$
     - $u_{Algebra} = 2$
5. Geometry:
   - Requirements vector:
     - $r_{Arithmetic} = 1$
   - Upskill vector:
     - $u_{Arithmetic} = 2$
     - $u_{Geometry} = 2$

### Transition Probabilities

- Probability of passing a module is proportional to the dot product of the student's skill level and the module's requirement vector. 
- If the student passes the module, their skills are updated by $s + \alpha \cdot r_i \cdot x$ where $x$ is a random soft mask and $\alpha > \beta$.
- If the student fails, they might still get a slight skill improvement $s + \beta \cdot r_i \cdot x$ where $\beta < \alpha$

### Terminal State Probability

Passing every module costs -1, and reaching the end state grants a reward of +1.  
The transition probability for passing a module can be represented as:  
$$P(s, s^ \prime, a_i) = \frac{1}{1+ \exp(-\gamma r_i \cdot s)} $$

where $a_i$ is the action of taking module with requirement $r_i$ and module learning outcomes $r_i$  

The state transition for passing the module:  
$$ s^ \prime = s + \alpha \cdot r_i \odot x $$

And for failing:  
$$ s^ \prime = s + \beta \cdot r_i \odot x $$

where $x$ is a mask that attenuates entries of $m$ by element wise multiplication $\odot$ . Each student has its own $x$ such that each student has its own learning abilities.

The terminal state probability involves the student reaching the desired minimal skill level in all domains, leading to a reward of +1.

### Rewards

The rewards for the agent will be based on a few factors:
- Passing or failing a module
- How long the agent is learning (cost of living)
- Achieving the requested goal

To incentivize the agent to find the most efficient route a cost of living is taking into account. We made the decision to not implement this through states (as a budget for example) but through reward logic. Implementing this feature through states will lead to a lot more different states to calculate which is something we do not want, especially when the functionality is the same.

## Algorithms

For the purpose of solving the problem we have implemented 3 different algorithms:

- Monte-Carlo,
- Q-Learning,
- Dynamic Programming.
<!-- add info about:
division of algorithms that know transition_probabilities and don't know
 "if np.random.rand() < epsilon or np.max(action_value_array[previous_state, :]) or best_action not in possible_actions:", that it helps the agent to start when action_value_array is full of zeros -->

### Monte-Carlo

It is controlled by the variable `monte_carlo_enabled`. During the episode of the Agent, a history of choices is recorded, and after the episode finishes the Action Value Function is evaluated, by analysing the Agent path from the end to the beginning. Algorithm inspects whether maximal difference between old and new value is smaller than threshold `theta`, which indicates that the optimal Action Value Function for given policy has been found. Process stops repeating when the Policy has Converged. Monte-Carlo Evaluation formula:

$$
Q({s}^m_t, {a}^m_t) \gets Q({s}^m_t, {a}^m_t) + \alpha ({g}^m_t - Q({s}^m_t, {a}^m_t))
$$

### Q-Learning

It is controlled by the variable `q_learning_enabled` and evaluates the Action Value Function every step of the agent. Algorithm inspects whether maximal difference between old and new value is smaller than threshold `theta`, which indicates that the optimal Action Value Function for given policy has been found. Process stops repeating when the Policy has Converged. Q-Learning Evaluation formula:

<!-- $$
action_value_array[previous_state, action] = (1 - alpha_q_learning) * action_value_array[
                previous_state, action] + alpha_q_learning * (reward + gamma * max(action_value_array[new_state, :]))
$$ -->

### Dynamic Programming

It is controlled by the variable `dynamic_programming_enabled` and does the Policy Evaluation and Policy Improvement to compute the Action Value Function based on the Greedy Policy with respect to Action Value Function, by implementation of the Bellman Equation:

$$
q_{\pi}(s,a)=\sum_{^{s^ \prime \in S}_{r \in R}} p(s^ \prime, r | s, a)[r + \gamma \sum_{a^ \prime \in A(s^ \prime)} \pi(a^ \prime | s^ \prime)q_{\pi}(s^ \prime | a^ \prime)]
$$

The $\sum_{a^ \prime \in A(s^ \prime)} \pi(a^ \prime | s^ \prime)q_{\pi}(s^ \prime | a^ \prime)$ inside the code is solved by usage of `action_value_array[new_state, np.argmax(action_value_array[new_state, :])]`, as the Greedy Policy chooses only one action, not any other, which means that in probabilities given by $\pi(a^ \prime | s^ \prime)$ they compose only of 0 and one 1, thus resulting in multiplying most of the values from the Action Value Function by 0. In order to omit unnecessary computation only the value of Action Value Function for the Action with probability of 1 is present.

#### Policy Evaluation

After each sweep through the Action Value Function the algorithm inspects whether maximal difference between old and new value is smaller than threshold `theta`, which indicates that the optimal Action Value Function for given policy has been found. Process stops repeating when the Policy has Converged.

#### Policy Improvement / Policy Extraction

Policy Improvement is done somewhat automatically, as the Policy is not stored, but is always calculated with the usage of `argmax()` function, which always returns Optimal Greedy Policy with respect to current Action Value Function.

### Conclusion

The model is incorporating the test data, and the agent is learning the optimal path to achieve the goal, but the model need further testing on more diverse scenarios. The ground basics for the project has been laid out.
