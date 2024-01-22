- If the alpha and beta parameter for calculating the new states are to close together the new state becomes the same state due to the values being rounded down. This leads to problems with the transition probabilitites. 

- Another problem we came across is the fact that the agent can take the same course multiple times to reach the skill goal. This is of course not something we want but it was a fun learning experience.

 - Theoretical discussion of how our project could be more interesting, focusing more on personalizing. Agent looks more at the will of the student not just give the best knowledge that leads to the best job for example.

- Offline q learning, learning from dynamic programming output

- How we interpret the upskill vector and rewards

- The upskill vectors do not get added litteraly they are a the maxiumum skill level you can achieve from said module. So s = 3 u = 5 then at most s can be 5 so not 3 + 5 = 8