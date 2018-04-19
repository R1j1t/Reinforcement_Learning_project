import gym
import numpy as np
import os
import random
from keras import backend
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Flatten, Dense, Conv2D
from collections import deque
from keras.optimizers import Adam, RMSprop
import json
# import visdom

class DQN:

    def __init__(self):
        # constructor 
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.num_episodes = 1000    # train from num_episodes number of episodes
        self.reward_sum = 0        # reward sum for current episode
        self.avg_reward_sum = 0
        self.minibatch_size = 32
        self.model = None    
        self.replay_memory = deque(maxlen = 2000)
        self.learning_rate = 0.001
        self.discount = 0.95
        # self.vis = visdom.Visdom()
        # self.x, self.y = [], []     

    # def memorize(astate, action, reward, bstate):
        
    def save_model(self, path = "model.h5"):
        self.model.save(path)
        # json_string = self.model.to_json()
        # with open(path, 'w') as outfile:
        #     json.dump(json_string, outfile)

    def init_model(self, load = False, path = "model.h5"):
        # load ConvNet 
        if(load):
            self.model = load_model(path)
            return

        self.model = Sequential()
        self.model.add(Dense(48, input_dim=4, activation='tanh'))
        self.model.add(Dense(48, activation='tanh'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))

    def train(self):
        env = gym.make('CartPole-v0')
        
        for epi in range(self.num_episodes):
            # obs = env.reset()
            # obs = self.preprocess(obs)
            # plt.imshow(obs, cmap = 'gray')
            # plt.show()
            # break

            print("<=====Episode - %d=====>"%(epi+1))
            self.reward_sum = 0
            done = False
            obs = env.reset()

            while(True):
                # store current  state in astate
                astate = np.reshape(obs, [1, 4])

                if(epi > 500):
                    env.render()    

                # action using epsilon-greedy 
                if(np.random.random() < self.epsilon):
                    action = np.random.choice(range(2))
                else:
                    action = np.argmax(self.model.predict(astate)[0])

                # next state + reward 
                obs, reward, done, info = env.step(action)
                bstate = np.reshape(obs, [1, 4])
                self.reward_sum = self.reward_sum + reward
                
                self.replay_memory.append((astate, action, reward, bstate, done))

                if(done):
                    print ("Reward for episode %d - %d"%(epi+1, self.reward_sum))
                    break
            #---------------------------------------------------------------------------------------#

            self.avg_reward_sum = (self.avg_reward_sum + self.reward_sum)

            # sample minibatch 
            if(len(self.replay_memory) > self.minibatch_size):
                minibatch = random.sample(self.replay_memory, self.minibatch_size)
            else:
                minibatch = self.replay_memory
            
            for astate, action, reward, bstate, done in minibatch:
                label_i = reward
                if not done:
                    label_i = (reward + self.discount * np.amax(self.model.predict(bstate)[0]))
                label = self.model.predict(astate)
                label[0][action] = label_i
                self.model.fit(astate, label, epochs = 1, verbose = 0)

            if(self.epsilon > 0.1):
                self.epsilon *= self.epsilon_decay

            # save model after every 100 episodes 
            if(epi % 100):
                self.save_model()

        	# # plot dynamic graph - reward_sum v/s episodes
         #    self.x.append(epi+1)
         #    self.y.append(self.reward_sum)
         #    trace = dict(x = self.x, y = self.y, mode = "lines", type = 'custom')
         #    layout = dict(title = "Reward v/s Episode", xaxis = {'title': 'Episode'}, yaxis = {'title': 'Reward'})
         #    self.vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})

dqn = DQN()
dqn.init_model()
dqn.train()
dqn.save_model()

# plt.figure()
# plt.plot(self.x2, self.y2)
# plt.show()
