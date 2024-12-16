Project files and implementations for CMPE591 (Deep Learning in Robotics course)

## To Run:















### Reward Plot History Records: 
added_target_network: 

The first value function couldn't learn at all because I didn't save the observation rewards at sample time, so the same network had to both predict the value and assess its own prediction, which causes the circularity problem encountered in offline RL algorithms like DQN. My thought at the time was to solve it the way it is solved in offline RL, by a target network, which partially solved the problem.


before_exploration:

I decided to incorporate an entropy reward term to encourage exploration at the time. This was the final version of the plot before I implemented that, which I saved to compare how well it does.

must_be_a_fluke:

The horrible result of me implementing the entropy reward term. I don't even remember how was it that horrible.

static_beta:

This is the result I fixed the problem with the entropy, using static coefficient for the learning rate. I wasn't satisfied with the improvement, so I wanted to try dynamic coefficient too, but then realized the problem was much deeper down so I postponed this.

after_rollout_buffer:

First output after I transitioned to training with states being in correct temporal order, instead of randomly sampling from the memory buffer like an offline algorithm. Honestly surprizing how long I have overlooked this. But the formula for value function is probably wrong, will update.
    
