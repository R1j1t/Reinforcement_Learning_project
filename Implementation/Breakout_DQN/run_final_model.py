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
from matplotlib import pyplot as plt

class DQN:

    def __init__(self):
        # constructor 
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.discount = 0.99
        self.learning_rate = 0.001
        self.num_episodes = 1000000    # train from num_episodes number of episodes
        self.reward_sum = 0        # reward sum for current episode
        self.history = np.zeros((82, 72, 4))    # 4 most recent frames    
        self.minibatch_size = 32
        self.model = None    
        self.replay_memory = deque(maxlen = 100000)
        self.action_map = {0: 2, 1: 3}
        self.learning_rate = 0.00025

    # def memorize(astate, action, reward, bstate):
        
    def save_model(self, path = "model.h5"):
        # self.model.save(path)
        # json_string = self.model.to_json()
        # with open(path, 'w') as outfile:
        #     json.dump(json_string, outfile)
        self.model.save("model.h5")

    def init_model(self, load = True, path = "model.h5"):
        # load ConvNet 
        if(load):
            self.model = load_model(path)
            return

        # define ConvNet 
        self.model = Sequential()
        
        # hidden layers 
        self.model.add(Conv2D(32, kernel_size = (8, 8), strides = (4, 4), activation = 'relu', input_shape = (82, 72, 4)))        
        self.model.add(Conv2D(64, kernel_size = (4, 4), strides = (2, 2), activation = 'relu'))
        self.model.add(Conv2D(64, kernel_size = (3, 3), strides = (1, 1), activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = 'relu'))
        
        self.model.add(Dense(4, activation = 'linear'))    # output layer 
        self.model.compile(RMSprop(self.learning_rate), loss='mean_squared_error')    # compile model 

    def phi_process(self, frame):
        # assigns frame as most recent frame in self.history 
        self.history[:, :, 0] = frame 
        self.history = np.roll(self.history, -1, axis = -1)
        # return self.history

    def preprocess(self, frame):
        # preprocess frame 
        frame = frame[32:195, 8:152, :]        # crop
        frame = frame[:, :, 0]        # remove color
        frame = frame[::2, ::2]
        frame[frame == 0] = 0
        frame[frame != 0] = 1
        return frame

    def train(self):
        env = gym.make('BreakoutDeterministic-v4')
        
        for epi in range(self.num_episodes):
            # testing crop params 
            #---------------------------------#
            # obs = env.reset()
            # obs = self.preprocess(obs)
            # plt.imshow(obs, cmap = 'gray')
            # plt.show()
            # break
            #---------------------------------#

            print("<=====Episode - %d=====>"%(epi+1))

            self.history = np.zeros((82, 72, 4))
            self.reward_sum = 0
            done = False
            obs = env.reset()
            obs = self.preprocess(obs)
            self.phi_process(obs)

            while(True):
                # store current  state in astate
                # astate = self.history

                env.render()    

                # if(len(self.replay_memory) < 50000):
                #     action = np.random.choice(env.action_space.n)
                #     # next state + reward 
                #     obs, reward, done, info = env.step(action)
                #     self.reward_sum = self.reward_sum + reward

                #     # process next state  
                #     obs = self.preprocess(obs)
                #     self.phi_process(obs)

                #     self.replay_memory.append((astate, action, reward, self.history, done))

                # else:
                # action using epsilon-greedy 
                # if(np.random.random() < self.epsilon):
                #     # action = env.action_space.sample()
                #     action = np.random.choice(env.action_space.n)
                # else:
                action = np.argmax(self.model.predict(np.expand_dims(self.history, axis = 0))[0])
                # action = env.action_space.sample()

                # next state + reward 
                obs, reward, done, info = env.step(action)
                self.reward_sum = self.reward_sum + reward

                # process next state  
                obs = self.preprocess(obs)
                self.phi_process(obs)

                # self.replay_memory.append((astate, action, reward, self.history, done))


                # # sample minibatch 
                # # if(len(self.replay_memory) > self.minibatch_size):
                # minibatch = random.sample(self.replay_memory, self.minibatch_size)
                # # else:
                # # minibatch = self.replay_memory
                
                # for astate, action, reward, bstate, done in minibatch:
                #     # reshape for matching input dimensions 
                #     astate = np.reshape(astate, (1,) + astate.shape)
                #     bstate = np.reshape(bstate, (1,) + bstate.shape)

                #     target = reward
                #     if not done:
                #         target = (reward + np.amax(self.model.predict(bstate)[0]))
                #     label = self.model.predict(astate)
                #     label[0][action] = target

                #     self.model.fit(astate, label, epochs = 1, verbose = 0)

                # if(self.epsilon > 0.1):
                #     self.epsilon *= self.epsilon_decay

                #---------------------------------------------------------------------------------------#

                if(done):
                    print ("Reward for episode %d - %d"%(epi+1, self.reward_sum))
                    break
            # save model after every 100 episodes 
            # if(epi % 100):
            #     self.save_model()
            

dqn = DQN()
dqn.init_model()
dqn.train()
dqn.save_model()
