import numpy as np
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import gym
import keras
from matplotlib import pyplot as plt
import os
from PIL import Image

gamma = 0.99
env = gym.make('Pong-v0')
input_dim = 80*80
action_space = 6  #for env.unwrapped.get_action_meanings() = 6, 1,2 are useful to us
train_x =[]
train_y=[]
total_utility = []
resume = True
observation = env.reset()
prev_frame = None
frames, proxy_labels,rewards = [],[],[] ## Required for batch training

# Defing the Neural network
model = Sequential()
model.add(Reshape((1,80,80),input_shape=(input_dim,)))
model.add(Conv2D(32,(9,9),subsample=(4, 4), border_mode='same',activation='relu',kernel_initializer = 'VarianceScaling'))
model.add(Flatten())
model.add(Dense(1000,activation = 'relu',kernel_initializer = 'VarianceScaling'))
model.add(Dense(100,activation = 'tanh',kernel_initializer = 'VarianceScaling'))
model.add(Dense(action_space,activation='softmax'))
model.compile(optimizer = keras.optimizers.Adam(lr=0.0001,amsgrad=True),
              loss = keras.losses.categorical_crossentropy)
if resume:
  model = keras.models.load_model('pong_model.h5')

def frame_preProcess(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

episode = 1
current_reward =0
utility = 0
rr=None

while True:
 #   env.render()
    current_frame = frame_preProcess(observation)
    difference_frame = current_frame - prev_frame if prev_frame is not None else np.zeros(input_dim)
    prev_frame = current_frame
 
    
    action_proba = model.predict(difference_frame.reshape([1,difference_frame.shape[0]])).flatten()
    #print(action_proba)
    
    ## Creating Batches for training
    frames.append(current_frame)
#    probs.append(action_proba)
    action = np.random.choice(action_space,1,p=action_proba)[0]
    y = np.zeros([action_space])
    y[action]=1
    proxy_labels.append(y-action_proba)

    ## get new information
    observation, reward, done, info = env.step(action)
    utility +=reward
    rewards.append(reward)

    if done:
        episode +=1
        #epx = np.vstack(frames)
        vstack_frames = np.vstack(frames)
        labels = np.vstack(proxy_labels)
        #print(proxy_labels)
        vstack_rewards = np.vstack(rewards)
        
        discounted_v_rewards = discount_rewards(vstack_rewards)
        discounted_v_rewards -= np.mean(discounted_v_rewards)
        #print('Before diving by std')
        #print(discounted_v_rewards)
        #discounted_v_rewards /= np.std(discounted_v_rewards)
        #print('Discounted rewards')
        #print(discounted_v_rewards)
        #print (labels,discounted_v_rewards)
        labels *= discounted_v_rewards
        #print('Labesl after multipling by discounted rewards')
        #print(labels)
        #print(discounted_v_rewards)
        

        train_x.append(frames)
        train_y.append(labels)
        frames,proxy_labels,rewards=[],[],[]
        
        if episode % 40== 0:
          print(action_proba)
 #         model.train_on_batch((np.vstack(train_x)),(np.vstack(train_y)))
          train_y,train_x = [],[]
          print('Training Done')
          
          
##          if episode % 100== 0:
##            os.remove('pong_model.h5') if os.path.exists('pong_model.h5') else None
##            model.save('pong_model.h5')
##   
##            os.remove('utility_data.npy') if os.path.exists('utility_data.npy') else None
##            np.save('utility_data',total_utility)
##         
##            print('Upload to Google Drive Complete')

          rr= utility if rr is None else rr*0.99 + utility*.01
          total_utility.append(utility)

          print (utility)
          utility = 0
          observation = env.reset()
          prev_frame = None

