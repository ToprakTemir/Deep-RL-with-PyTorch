### Reward Plot History Records: 
added_target_network: 

The first value function couldn't learn at all because I didn't save the observation rewards at sample time, so the same network had to both predict the value and assess its own prediction, which causes the circularity problem encountered in offline RL algorithms like DQN. My thought at the time was to solve it the way it is solved in offline RL, by a target network, which partially solved the problem.


before_exploration:

I decided to incorporate an entropy reward term to encourage exploration at the time. This was the final version of the plot before I implemented that, which I saved to compare how well it does.



static_beta:

This is the result I fixed the problem with the entropy, using static coefficient for the learning rate. I wasn't satisfied with the improvement, so I wanted to try dynamic coefficient too, but then realized the problem was much deeper down so I postponed this.


after_rollout_buffer:

First output after I transitioned to training with states being in correct temporal order, instead of randomly sampling from the memory buffer like an offline algorithm. Honestly surprising how long I have overlooked this. But the formula for value function is probably wrong, I will continue working on it.


successful_gae:

Finally, after completing the Generalized Advantage Estimation (GAE) in the rollout buffer and fixing all the problems I caused at the start because of the way I didn't understand it completely at the start, this is the first result that I was able to break the previous plateau.
The value function seems to work good, GAE might have actually finalized it
Future work: 
- The policy function's entropy is VERY slowly but surely increasing, meaning that it is not first increasing to satisfy the exploration reward and then decreasing after it finds a robust good policy. It just very slowly increases (it doubled first in 2000 epochs, and then in 5000 more epochs) That might be looked into.
- Related to the above issue, maybe look into more exploration strategies even though they are not implemented in stable baselines's implementation
- Hyperparameter tuning
- Creating a gymnasium environment to test on stable baseline's ppo implementation 
