# Reinforcement_Learning_project

This repository holds our Reinforcement learning implementation for 2 vary popular game namely [Pong](https://en.wikipedia.org/wiki/Pong#Gameplay) and [cartpole](https://gym.openai.com/envs/CartPole-v1/). We have used Keras for model building. Two different types of algorithms are used to play the game, the first was Deep Q learning and second was Policy gradient. They both have there advantages and disadvantages. You can read more about them on wikipedia or research papers (preferred option).

All the algorithms were trained on Nvdia GPU. The results shown below do mention the training time as well. To differentiate the two algorithms we have used the following conventions.


-  Game_DQN = implementation of "Game" by Deep Q-learning. This implementation is based on Natures paper on Deep Q learning.

-  Game_pg = implementation of "Game" by policy gradient. This code is based on Andrej Karpathy blog on policy gradient.

### Results for cartpole

<p align="center">
<img src="https://github.com/R1j1t/Reinforcement_Learning_project/blob/master/Implementation/CartPole_PG/cartpole_pg.gif">
</p>
<p align="center">
Model training took less than 5 mins on GPU
</p>

### Trained model for Pong

<p align="center">
<img src="https://github.com/R1j1t/Reinforcement_Learning_project/blob/master/Implementation/Pong_PG/Pong_pg.gif">
</p>
</p>
<p align="center">
These results were achieved after 6-7 hrs of  Training on GPU
</p>

### Dependencies:
- Keras (2.1.5+)
- OpenAI Gym (0.10.3)
- Python (3.5+)

### Further Reading
1. Reinforcement Learning: An Introduction by Richard S. Sutton and Andrew G. Barto
1. [Andrej Karpathy blog on policy gradient](https://karpathy.github.io/2016/05/31/rl/)
2. [Deep Q learning paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
