
![alt text](image.png)

Architecture:-

1.The setup requires two networks, one for parameterized representation of the policy and he other for the for the Value estimates of the state

2.The network has total 3 layers.Instead of having 2 separate networks to learn policy and value, The first two layers (input layers and hidden layers) are merged while keeping the final layer with two separate heads

3.The network has one hidden layer.Input dimension is as same as the dimension of state space and it has 2 output heads, one for the policy with the dimension of action space and the other to represent value of the state

Actor Critic methods:-

Here we start by considering the objective function to maximize the estimated Reward while interacting with the stochastic environment. Then we move along the direction of the gradient of the reward estimate function.
The Monte Carlo estimates as the reward function, leads to REINFORCE, But the need to reduce the variance is the primary motive to Actor Critic method where we reduce the policy variance by learning the estimated value of the state parallely
