# Group Project

Grzegorz Malisz & Tiemon Steeghs

[GitHub Repository](https://github.com/grzgm/deth-group-project)

## Context

We are focusing on creating a model that would ease and enchance the process of learning from the MOOCs [Massive Open Online Courses]. The main goal of the model is to find the best learning path for user, based on user's current profile and his goal. The path established by the agent should minimalise the dropout posibility, learnign curve and maximalise the speed of learning. For our project we have picked the domain of Math MOOCs, but our solution can be extended to other topics.

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
9. Differential Equations

Discriptive visualisation of the state: $(Arithmetic, Geometry, Algebra, Calculus, Statistics, Discrete Mathematics, Logic, Mathematical Analysis, Differential Equations)$

Later in the paper we will use the two notation, which ease understanding and writing while serving different purposes. For example, to refer to the Arithmetic skill level we will use $s_1$, as well as $s_{Arithmetic}$ notation.

### Actions

Actions are defined by taking specific modules, each module has an associated upskilling vector $r_i$ and requirement vector $u_i$. Note that if value of the certain skill in the vector is 0, it is not denoted for the sake of clarity.

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
