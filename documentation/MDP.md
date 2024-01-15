# Group Project

Grzegorz Malisz & Tiemon Steeghs

[GitHub Repository](https://github.com/grzgm/deth-group-project)

## Context

We are focusing on creating a model that would ease and enchance the process of learning from  MOOCs [Massive Open Online Courses]. The main goal of the model is to find the best learning path for user, based on user's current profile and his goal. The path established by the agent should minimalise the dropout posibility, learning curve and maximalise the speed of learning. For our project we have picked the domain of Math MOOCs, but our solution can be extended to other topics.

For our implementation of our model we use the Markov Decision Process (MDP) framework. A full definition of our MDP can be found further below.
## Math MOOC

For the development purpose we have used this sturcture of MOOC:

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
5. Statistics
6. Discrete Mathematics
7. Logic
8. Mathematical Analysis

Discriptive visualisation of the state: $(Arithmetic, Geometry, Algebra, Calculus, Statistics, Discrete Mathematics, Logic, Mathematical Analysis)$

Later in the paper we will use the two notation, which ease understanding and writing while serving different purposes. For example, to refer to the Arithmetic skill level we will use $s_1$, as well as $s_{Arithmetic}$ notation.

### Actions

Actions are defined by taking specific modules, each module has an associated upskilling vector $u_i$ and requirement vector $r_i$. Note that if value of the certain skill in the vector is 0, it is not denoted for the sake of clarity.

#### Combo Modules
1. **Statistical Methods in Number Theory for Beginners**
	-  Requirements vector:
		- $r_{Arithmetic} = 1$
		- $r_{Statistics} = 1$
	- Upskill vector:
		- $u_{Arithmetic} = 2$
		- $u_{Statistics} = 2$
2. **Algebraic Statistics**
	-  Requirements vector:
		- $r_{Algebra} = 2$
		- $r_{Statistics} = 3$
	- Upskill vector:
		- $u_{Algebra} = 4$
		- $u_{Statistics} = 4$
3. **Discrete Geometry and Logic**
	- Requirements vector:
		- $r_{Algebra} = 2$
		- $r_{Geometry} = 2$
		- $r_{Discrete Mathematics} = 2$
		- $r_{Logic} = 2$
	- Upskill vector:
		- $u_{DiscreteMathematics} = 3$
		- $u_{Logic} = 3$
#### Arithmetic modules:
1. **Basic Arithmetic**
	- Requirements vector:
		- None
	- Upskill vector:
		- $u_{Arithmetic} = 3$
		- $u_{Geometry} = 1$
		- $u_{Algebra} = 1$
		- $u_{Calculus} = 1$

#### Algebra Modules
1. **Elementary Algebra 
	- Requirements vector:
		- None
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Geometry} = 1$
		- $u_{Algebra} = 2$
2. **Intermediate Algebra:**
    - - Requirements vector:
		- $r_{Algebra} = 2$
		- $r_{Arithmetic} = 1$
		- $r_{Geometry} = 1$
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Algebra} = 3$
3. **Linear Algebra:**
    - Requirements vector:
		- $r_{Algebra} = 5$
		- $r_{Arithmetic} = 3$
		- $r_{Geometry} = 3$
		- $r_{Calculus} = 2$
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Geometry} = 1$
		- $u_{Algebra} = 3$

#### Geometry modules 
1. **Geometry:**
	- Requirements vector:
		- $r_{Arithmetic} = 1$
	- Upskill vector:
		- $u_{Arithmetic} = 2$
		- $u_{Geometry} = 2$
2. **Trigonometry:**
    - Requirements vector:
		- $r_{Arithmetic} = 3$
		- $r_{Geometry} = 2$
	- Upskill vector:
		- $u_{Arithmetic} = 2$
		- $u_{Geometry} = 3$

#### Calculus modules
1. **Pre-Calculus:**
    - Integrate algebra, geometry, and trigonometry to prepare for the study of calculus. Topics may include functions, limits, and basic mathematical modeling.
    - Requirements vector:
		- $r_{Arithmetic} = 1$
		- $r_{Geometry} = 1$
		- $r_{Algebra} = 1$
	- Upskill vector:
		- $u_{Arithmetic} = 2$
		- $u_{Calculus} = 2$
		- $u_{Geometry} = 2$
		- $u_{Algebra} = 1$
2. **Calculus:**
    - Start with differential calculus, covering concepts like limits, derivatives, and applications. Then progress to integral calculus, exploring integrals and their applications.
    - Requirements vector:
		- $r_{Arithmetic} = 2$
		- $r_{Geometry} = 2$
		- $r_{Algebra} = 2$
		- $r_{Calculus} = 2$
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Calculus} = 3$
		- $u_{Geometry} = 1$
		- $u_{Algebra} = 1$
3. **Differential Equations:**
    - Study ordinary and partial differential equations and their applications in modeling real-world phenomena.
    - Requirements vector:
		- $r_{Arithmetic} = 5$
		- $r_{Geometry} = 3$
		- $r_{Algebra} = 3$
		- $r_{Calculus} = 5$
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Calculus} = 4$
		- $u_{Geometry} = 1$
		- $u_{Algebra} = 1$

#### Statistics modules
1. Probability and Statistics
	- Understand the principles of probability theory and statistical analysis. This is crucial for data analysis and decision-making in various fields.
	- Requirements vector:
		- $r_{Algebra} = 2$
		- $r_{Arithmetic} = 2$
		- $r_{Statistics} = 1$
	- Upskill vector:
		- $u_{Arithmetic} = 1$
		- $u_{Statistics} = 3$



### Transition Probability Function

Probability of passing a module is proportional to the dot product of the student's skill level and the module's requirement vector. - If the student passes the module, their skills are updated by $s + \alpha \cdot r_i \cdot x$ where $x$ is a random soft mask and $\alpha > \beta$.

- If the student fails, they might still get a slight skill improvement $s + \beta \cdot r_i \cdot x$ where $\beta < \alpha$

### Terminal State Probability

Passing every module costs -1, and reaching the end state grants a reward of +1.  
The transition probability for passing a module can be represented as:  
$$P(s, s', a_i) = \frac{1}{1+ \exp(-\gamma u_i \cdot s)} $$

where $a_i$ is the action of taking module with requirement $u_i$ and module learning outcomes $r_i$  
The state transition for passing the module:  
$$ s' = s + \alpha \cdot r_i \odot x $$

And for failing:  
$$ s' = s + \beta \cdot r_i \odot x $$

where $x$ is a mask that attenuates entries of $m$ by element wise multiplication $\odot$ . Each student has its own $x$ such that each student has its own learning abilities.

The terminal state probability involves the student reaching the desired minimal skill level in all domains, leading to a reward of +1.

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