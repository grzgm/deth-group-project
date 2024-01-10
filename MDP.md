For our MDP we have picked the domain Math our MDP defines the different skills that a mathematician can learn and we have defined modules which the agent can take te learn said skills. test
## States

 $(s_1, s_2, \ldots, s_N)$ : Skill levels in various domains.

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

So states = $(Arithmetic, Geometry, Algebra, Calculus, Statistics, Discrete Mathematics, Logic, Mathematical Analysis, Differential Equations)$
## Actions

- Actions are defined by taking specific modules, each module has an associated upskilling vector $m_i$  and requirement vector $Q_i$.

#### Combo Modules
1. Statistical Methods in Number Theory for Beginners
	- Skills: Arithmetic & Statistics
	- Requirements vector: (1, 0, 1, 0, 1, 0, 0, 0, 0)
	- Upskill vector: (2, 0, 1, 0, 2, 0, 0, 0, 0)
2. Algebraic Statistics
	- Skills: Algebra & Statistics
	- Requirements vector: (4, 3, 3, 1, 2, 0, 0, 0, 0, 0)
	- Upskilling vector: (1, 1, 2, 0, 3, 0, 0, 0, 0)
3. Discrete Geometry and Logic
	- Skills: Discrete Geometry &  Logic
#### Arithmetic modules:
1. **Basic Arithmetic**
	- Develop a strong foundation in basic arithmetic operations, including addition, subtraction, multiplication, and division.
	- Requirements vector: $(0, 0, 0, 0, 0, 0, 0, 0, 0,)$
	- Upskilling vector: $(3,1,1,1,0,0,0,0,0)$

#### Algebra Modules
1. **Elementary Algebra 
	- Learn algebraic concepts such as variables, equations, inequalities, and basic operations with polynomials.
	- Requirements vector: (0, 0, 0, 0, 0, 0, 0, 0, 0,)
		- Upskilling vector: (1, 1, 2, 0, 0, 0, 0, 0, 0) 
2. **Intermediate Algebra:**
    - Delve deeper into algebraic concepts, including factoring, rational expressions, and systems of equations.
3. **Linear Algebra:**
    - Learn about vector spaces, matrices, linear transformations, and eigenvalues. Linear algebra provides a foundation for various advanced mathematical topics.

#### Geometry modules 
1. **Geometry:**
    - Explore Euclidean geometry, which includes the study of shapes, angles, lines, and geometric constructions.
    - Requirements vector: (1, 0, 0, 0, 0, 0, 0, 0, 0)
    - Upskilling vector: (1, 0, 0, 0, 0, 0, 0, 0, 0)
2. **Trigonometry:**
    - Understand trigonometric functions, identities, and applications, especially in the context of right-angled triangles.

#### Calculus modules
1. **Pre-Calculus:**
    - Integrate algebra, geometry, and trigonometry to prepare for the study of calculus. Topics may include functions, limits, and basic mathematical modeling.
2. **Calculus:**
    - Start with differential calculus, covering concepts like limits, derivatives, and applications. Then progress to integral calculus, exploring integrals and their applications.
3. **Differential Equations:**
    - Study ordinary and partial differential equations and their applications in modeling real-world phenomena.
    - Requirement  vector:  $(3, 3, 3, 3, 0, 3, 1, 3, 3)$
#### Statistics modules
1. Probability and Statistics
	- Understand the principles of probability theory and statistical analysis. This is crucial for data analysis and decision-making in various fields.
### Transition Probability Function:
- Probability of passing a module is proportional to the dot product of the student's skill level and the module's requirement vector.
- If the student passes the module, their skills are updated by $s + \alpha \cdot m_i \cdot x$ where  $x$ is a random soft mask and $\alpha > \beta$.
- If the student fails, they might still get a slight skill improvement  $s + \beta \cdot m_i \cdot x$ where $\beta < \alpha$

### Terminal State Probability:
- Passing every module costs -1, and reaching the end state grants a reward of +1.

The transition probability for passing a module can be represented as: 
$$P(s, s', a_i) = \frac{1}{1+ \exp(-\gamma Q_i \cdot s)} $$

where $a_i$ is the action of taking module with requirement $Q_i$ and module learning outcomes $m_i$  
The state transition for passing the module: 
$$ s' = s + \alpha \cdot m_i \odot x $$

And for failing:
$$ s' = s + \beta \cdot m_i \odot x $$

where $x$ is a mask that attenuates entries of $m$ by element wise multiplication $\odot$ . Each student has its own $x$  such  that each student has  its own learning abilities.

The terminal state probability involves the student reaching the desired minimal skill level in all domains, leading to a reward of +1.
