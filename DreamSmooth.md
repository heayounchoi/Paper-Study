# DreamSmooth: Improving Model-based Reinforcement Learning via Reward Smoothing
[Introduction]
- humans often plan actions with a rough estimate of future rewards, instead of the exact reward at the exact moment
- predicting the exact reward is often challenging since it can be ambiguous, delayed, or not observable
- Inspired by the human intuition that only a rough estimate of rewards is sufficient, DreamSmooth learns to predict a temporally-smoothed reward rather than the exact reward at each timestep
- this technique is especially beneficial in environments with the following characteristics: sparse rewards, partial observability, and stochastic rewards

[Related Work]
- Model-based reinforcement learning (MBRL) leverages a dynamics model of an environment and a reward model of a desired task to plan a sequence of actions that maximaize the total reward
- with the dynamics and reward models, an agent can simulate a large number of candidate behaviors in imagination instead of in the physical environment
- compared to the efforts on learning a better world model, learning an accurate reward model has been largely overlooked
- reward prediction is strongly correlated to task performance when trained on an offline dataset, while limited to dense-reward environments
- this paper proposes a simple method to improve reward prediction in MBRL

[Approach]
- The main goal of this paper is to understand how challenging reward prediction is in MBRL and propose reward smoothing, which makes reward prediction easier to learn
