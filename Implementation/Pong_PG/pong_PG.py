import numpy as np
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import gym
import keras
import os

### This only needs to be done once in a notebook.##
##!pip install -U -q PyDrive
##from pydrive.auth import GoogleAuth
##from pydrive.drive import GoogleDrive
##from google.colab import auth
##from oauth2client.client import GoogleCredentials
##
### Authenticate and create the PyDrive client.
### This only needs to be done once in a notebook.
##auth.authenticate_user()
##gauth = GoogleAuth()
##gauth.credentials = GoogleCredentials.get_application_default()
##drive = GoogleDrive(gauth)


class Solve:

    def __init__(self):

        self.gamma = 0.99
        self.env = gym.make('Pong-v0')
        self.learning_rate = .01
        self.input_dim = 80*80
        self.action_space = 6  #for env.unwrapped.get_action_meanings() = 6, 1,2 are useful to us
        self.train_x =[]
        self.train_y=[]
        self.resume =True
        self.prev_frame = None
        self.lisframes, self.proxy_labels, self.rewards = [],[],[]
        
    ##refining the frame
    def refiningframe(self, Image):
     
      Image = Image[35:195] 
      Image = Image[::2,::2,0] 
      Image[Image == 144] = 0 
      Image[Image == 109] = 0 
      Image[Image!= 0] = 1
      
      return Image.astype(np.float).ravel()
    ##getting the reward array

    def reWards(self, ar):
      discounted_r = np.zeros_like(ar)
      d1 = 0
      for t in reversed(range(0, ar.size)):
        if ar[t] != 0: d1 = 0
        d1 = d1 * self.gamma + ar[t]
        discounted_r[t] = d1
      return discounted_r

    def init_model(self):

        # Defing the Neural network
##        self.model = Sequential()
##        self.model.add(Reshape((1,80,80),input_shape=(self.input_dim,)))
##        self.model.add(Conv2D(32,(9,9),subsample=(4, 4), border_mode='same',activation='relu',kernel_initializer = 'VarianceScaling'))
##        self.model.add(Flatten())
##        self.model.add(Dense(10,activation = 'relu'))
##        self.model.add(Dense(self.action_space,activation='softmax'))
##        self.model.compile(optimizer = keras.optimizers.Adam(lr=0.001),
##                      loss = keras.losses.categorical_crossentropy)
        if self.resume:
          self.model = keras.models.load_model('pong_model.h5')

    ##training the model
    def train(self):
        episode = 1
        current_reward =0
        utility = 0
        rr=None
        observation = self.env.reset()

        while True:
        #    env.render()
            recentframe = self.refiningframe(observation)
            difference_frame = recentframe - self.prev_frame if self.prev_frame is not None else np.zeros(self.input_dim)
            self.prev_frame = recentframe
            action_proba = self.model.predict(difference_frame.reshape([1,difference_frame.shape[0]])).flatten()

            #print (action_proba)
            ## Creating Batches for training
            self.lisframes.append(difference_frame)
           
            action = np.random.choice(self.action_space,1,p=action_proba)[0]
            y = np.zeros([self.action_space])
            y[action]=1
            self.proxy_labels.append(y-action_proba)

            observation, reward, done, info = self.env.step(action)
            utility +=reward
            self.rewards.append(reward)
            ## if episode is completed

            if done:
                episode +=1
                print(utility)
                
                vstack_lisframes = np.vstack(self.lisframes)
            
                labels = np.vstack(self.proxy_labels)
                vstack_rewards = np.vstack(self.rewards)
                
                disvrewards = self.reWards(vstack_rewards)
                disvrewards -= np.mean(disvrewards)
                
                labels *= disvrewards

                self.train_x.append(self.lisframes)
                self.train_y.append(labels)
                self.lisframes, self.proxy_labels, self.rewards=[],[],[]
                ## training after 30 episodes
                if episode % 30 == 0:
                  y_train = np.squeeze(np.vstack(self.train_y))
             
                  self.model.train_on_batch(np.squeeze(np.vstack(self.train_x)), y_train)
                  
                  self.train_y,self.train_x = [],[]
                  ##saving the model

                  os.remove('pong_model.h5') if os.path.exists('pong_model.h5') else None
                  self.model.save('pong_model.h5')

                  rr= utility if rr is None else rr*0.99 + utility*.01
                  print("episode number %d"%episode)
                  print(utility)
                  rr = 0

                utility = 0
                observation = self.env.reset()
                self.prev_frame = None
                  

solve = Solve()
solve.init_model()
solve.train()
