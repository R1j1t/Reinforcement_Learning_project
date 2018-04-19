import numpy as np
from keras.layers import Input, Dense, Reshape
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import gym
import keras
import os,visdom

 
gamma = 0.99
env = gym.make('CartPole-v1')
input_dim = 4
action_space = 2  #for env.unwrapped.get_action_meanings() = 6, 1,2 are useful to us
train_x =[]
train_y=[]
total_utility = []
resume = False
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
model.add(Dense(10,activation = 'relu',input_shape=(input_dim,)))
model.add(Dense(8,activation = 'relu'))
model.add(Dense(action_space,activation='softmax'))
model.compile(optimizer = keras.optimizers.adam(lr=0.001),
              loss = keras.losses.categorical_crossentropy)
if resume:
  model = keras.models.load_model('pong_model_2.h5')


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
    action = np.random.choice(2,1,p=action_proba)[0]
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
        utility = 0    
        prev_frame = None
        episode +=1
        #epx = np.vstack(frames)
        vstack_frames = np.vstack(frames)
        labels = np.vstack(proxy_labels)
        vstack_rewards = np.vstack(rewards)
 #       frames,proxy_labels,rewards=[],[],[]
        #print(vstack_frames.shape,labels.shape,vstack_rewards.shape)
        
        discounted_v_rewards = discount_rewards(vstack_rewards)
##        discounted_v_rewards -= np.mean(discounted_v_rewards)
##        discounted_v_rewards /= np.std(discounted_v_rewards)
        labels *= discounted_v_rewards
        #print (labels)

        train_x.append(frames)
        train_y.append(labels)
        frames,proxy_labels,rewards=[],[],[]
        #print (utility,episode)
      

        #print (len(train_x),len(train_y))
        
        if episode % 20== 0:
          #print (len(probs),np.squeeze(np.vstack(train_y)).shape)
          
          model.train_on_batch((np.vstack(train_x)),(np.vstack(train_y)))
          train_y,train_x,probs = [],[],[]
          
##          
##          if utility > 100:
##          
####            os.remove('pong_model.h5') if os.path.exists('pong_model.h5') else None
####            model.save('pong_model.h5')
####            file = drive.CreateFile({'parents':[{u'id': '1GYGA24nNScPpiZTq9HX3e-bVERebRyxg'}]})
####            file.SetContentFile('{}.h5'.format(str('pong_model')))
####            file.Upload()
##       
##            os.remove('utility_data.npy') if os.path.exists('utility_data.npy') else None
##            np.save('utility_data',total_utility)
##            file = drive.CreateFile({'parents':[{u'id': '1Lfg0GyjEywGXUKXK2Wg_RueEFO91RlAg'}]})
##            file.SetContentFile('{}.npy'.format(str('utility_data')))
##            file.Upload()
##              rr= utility if rr is None else rr*0.99 + utility*.01
##              total_utility.append(utility)

              #print (utility)
              #utility = 0
 
##        vis = visdom.Visdom()
##        xx.append(episode)
##        yy.append(utility)
##        trace = dict(x = xx, y = yy, mode = "lines", type = 'custom')
##        layout = dict(title = "Reward v/s Episode", xaxis = {'title': 'Episode'}, yaxis = {'title': 'Reward'})
##        vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
