import numpy as np
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import gym
import keras
import os

 
gamma = 0.99
env = gym.make('CartPole-v1')
input_dim = 4
action_space = 2  #for env.unwrapped.get_action_meanings() = 6, 1,2 are useful to us
train_x =[]
train_y=[]
total_utility = []
resume = True
observation = env.reset()
prev_frame = None
frames, proxy_labels,rewards,probs = [],[],[],[] ## Required for batch training
xx=[]
yy=[]

# Defing the Neural network
model = Sequential()
#model.add(Reshape((1,80,80),input_shape=(input_dim,)))
#model.add(Conv2D(32,(9,9),subsample=(4, 4), border_mode='same',activation='relu',kernel_initializer = 'VarianceScaling'))
#model.add(Flatten())
model.add(Dense(8,activation = 'relu',input_shape=(input_dim,)))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(action_space,activation='softmax'))
model.compile(optimizer = keras.optimizers.adam(lr=0.01),
              loss = keras.losses.categorical_crossentropy)
if resume:
  model = keras.models.load_model('cartpole_model.h5')


## Preprocessing the data
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
##    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

episode = 0
current_reward =0
utility = 0
rr=None


while True:
    env.render()
#    observation = frame_preProcess(observation)
    #difference_frame = current_frame - prev_frame if prev_frame is not None else np.zeros(input_dim)
    #prev_frame = current_frame
    action_proba = model.predict(observation.reshape([1,observation.shape[0]])).flatten()
    probs.append(action_proba)
    ## Creating Batches for training
    frames.append(observation)
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
        observation = env.reset()
        print(utility)

        episode +=1
        #epx = np.vstack(frames)
        vstack_frames = np.vstack(frames)
        labels = np.vstack(proxy_labels)
        vstack_rewards = np.vstack(rewards)
 #       frames,proxy_labels,rewards=[],[],[]
        #print(vstack_frames.shape,labels.shape,vstack_rewards.shape)
        
        discounted_v_rewards = discount_rewards(vstack_rewards)
        discounted_v_rewards -= np.mean(discounted_v_rewards)
        discounted_v_rewards /= np.std(discounted_v_rewards)
        labels *= discounted_v_rewards
        #print (labels)

        train_x.append(frames)
        train_y.append(labels)
        frames,proxy_labels,rewards=[],[],[]
        

        #print (len(train_x),len(train_y))
        
        if episode % 5== 0:
          #print (len(probs),np.squeeze(np.vstack(train_y)).shape)
          
          y_train = probs + 0.001 * np.squeeze(np.vstack(train_y))
          model.train_on_batch((np.vstack(train_x)),(np.vstack(y_train)))
          train_y,train_x,probs = [],[],[]
          
          
          if utility > 100:
            print (utility,episode)
          
            os.remove('cartpole_model.h5') if os.path.exists('cartpole_model.h5') else None
            model.save('cartpole_model.h5')

        prev_frame = None
        utility = 0    

