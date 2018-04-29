import tensorflow as tf
import gym
import random
import math
from collections import deque
import numpy as np
import pdb

class Actor(object):
    """docstring for Actor"""
    def __init__(self,environmentSpace,actionSpace):
        self.graph        = tf.Graph()
        self.sess         = tf.InteractiveSession()
        self.envSize      = environmentSpace
        self.actionSize   = actionSpace
        self.state_i      = tf.placeholder("float",[None,self.envSize])
        self.dq_da        = tf.placeholder("float",[None,self.actionSize])
        self.theta_u      = []
        self.batchSize    = 64
        self.theta_u_t    = None
        self.learningRate = 1e-4 # Tune this
        self.tau          = 1e-2 # Tune this
        self.actorNetwork()
        self.gradients    = tf.gradients(self.actorNet,self.theta_u,-self.dq_da)#/self.batchSize
        self.optimizer    = tf.train.AdamOptimizer(self.learningRate).apply_gradients(zip(self.gradients,self.theta_u))
        self.sess.run(tf.global_variables_initializer())
        
    def actorNetwork(self):
        hiddenLayers = [400,300]
        numberOfLayers = len(hiddenLayers)+1
        outputLayerSize = 0
        for i in range(numberOfLayers):
            inputLayerSize = outputLayerSize
            
            if i == 0:
                inputLayerSize = self.envSize
                h1 = self.state_i
                h1_t = self.state_i
            
            
            if i == numberOfLayers-1:
                outputLayerSize = self.actionSize
                std = 0.003
            else:
                outputLayerSize =  hiddenLayers[i]
                std = 1/math.sqrt(inputLayerSize)

            setattr(self,'w'+str(i+1),tf.Variable(tf.random_uniform([inputLayerSize,outputLayerSize],-std,std)))
            setattr(self,'w_t'+str(i+1),getattr(self,'w'+str(i+1)))      
            setattr(self,'b'+str(i+1),tf.Variable(tf.random_uniform([outputLayerSize],-std,std)))   
            setattr(self,'b_t'+str(i+1),getattr(self,'b'+str(i+1)))
            self.theta_u.append(getattr(self,'w'+str(i+1)))
            self.theta_u.append(getattr(self,'b'+str(i+1)))

            if i < numberOfLayers-1:
                h1 = tf.nn.softplus(tf.matmul(h1,getattr(self,'w'+str(i+1)))+getattr(self,'b'+str(i+1)))
                h1_t = tf.nn.softplus(tf.matmul(h1_t,getattr(self,'w_t'+str(i+1)))+getattr(self,'b_t'+str(i+1)))
            
            else:
                setattr(self,'actorNet',tf.tanh(tf.matmul(h1,getattr(self,'w'+str(i+1)))+getattr(self,'b'+str(i+1)))) # doubtful as it bounds the actions to +1 to -1. Should we scale?
                setattr(self,'actorNet_t',tf.tanh(tf.matmul(h1_t,getattr(self,'w_t'+str(i+1)))+getattr(self,'b_t'+str(i+1)))) # doubtful as it bounds the actions to +1 to -1. Should we scale?
        
    def updateActor(self,state,dq_da):
       self.sess.run(self.optimizer,feed_dict={self.state_i:state,self.dq_da:dq_da})       

    def getAction(self,state):
        return self.sess.run(self.actorNet,feed_dict={self.state_i:state})       

    def getTargetAction(self,state):
        return self.sess.run(self.actorNet_t,feed_dict={self.state_i:state})

    def updateTargetActor(self):
        for i in range(int(len(self.theta_u)/2)):
            self.sess.run(getattr(self,'w_t'+str(i+1)).assign(self.tau*getattr(self,'w'+str(i+1))+getattr(self,'w_t'+str(i+1))*(1-self.tau)))  
            self.sess.run(getattr(self,'b_t'+str(i+1)).assign(self.tau*getattr(self,'b'+str(i+1))+getattr(self,'b_t'+str(i+1))*(1-self.tau)))   

class Critic(object):
    """docstring for Critic"""
    def __init__(self,environmentSpace,actionSpace):
        self.graph        = tf.Graph()
        self.sess         = tf.InteractiveSession()
        self.envSize      = environmentSpace
        self.actionSize   = actionSpace
        self.state_i      = tf.placeholder("float",[None,self.envSize])
        self.action_i     = tf.placeholder("float",[None,self.actionSize])
        self.theta_q      = []
        self.theta_q_t    = None
        self.learningRate = 1e-3 # Tune this
        self.batchSize    = 64
        self.tau          = 1e-2 # Tune this
        self.criticNetwork()
        self.y_i          = tf.placeholder("float",[None,1])
        self.cost         = tf.pow(self.y_i-self.criticNet,2)/self.batchSize
        self.optimizer    = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
        self.gradients    = tf.gradients(self.criticNet, self.action_i)
        self.gradients    = [self.gradients[0]/tf.to_float(tf.shape(self.gradients[0])[0])]
        self.sess.run(tf.global_variables_initializer())
        
    def criticNetwork(self):
        hiddenLayers = [400,300]
        numberOfLayers = len(hiddenLayers)+1
        outputLayerSize = 0
        for i in range(numberOfLayers):
            inputLayerSize = outputLayerSize
            
            if i == 0:
                inputLayerSize = self.envSize
                h1 = self.state_i
                h1_t = self.state_i
            
            if i == numberOfLayers-1:
                outputLayerSize = 1
                std = 0.0003
            else:
                outputLayerSize =  hiddenLayers[i]
                std = 1/math.sqrt(inputLayerSize)

            setattr(self,'w'+str(i+1),tf.Variable(tf.random_uniform([inputLayerSize,outputLayerSize],-std,std)))
            setattr(self,'w_t'+str(i+1),getattr(self,'w'+str(i+1)))      
            setattr(self,'b'+str(i+1),tf.Variable(tf.random_uniform([outputLayerSize],-std,std)))   
            setattr(self,'b_t'+str(i+1),getattr(self,'b'+str(i+1))) 
            self.theta_q.append(getattr(self,'w'+str(i+1)))
            self.theta_q.append(getattr(self,'b'+str(i+1)))

            if i == 1:
                setattr(self,'w'+str(i+1)+'_a',tf.Variable(tf.random_uniform([self.actionSize,outputLayerSize],-std,std)))
                setattr(self,'w_t'+str(i+1)+'_a',getattr(self,'w'+str(i+1)+'_a'))      
                
            if i < numberOfLayers-1 and i!=1:
                h1 = tf.nn.softplus(tf.matmul(h1,getattr(self,'w'+str(i+1)))+getattr(self,'b'+str(i+1)))
                h1_t = tf.nn.softplus(tf.matmul(h1_t,getattr(self,'w_t'+str(i+1)))+getattr(self,'b_t'+str(i+1)))
            elif i == 1:
                h1 = tf.nn.softplus(tf.matmul(h1,getattr(self,'w'+str(i+1)))+tf.matmul(self.action_i,getattr(self,'w'+str(i+1)+'_a'))+getattr(self,'b'+str(i+1)))
                h1_t = tf.nn.softplus(tf.matmul(h1_t,getattr(self,'w_t'+str(i+1)))+tf.matmul(self.action_i,getattr(self,'w_t'+str(i+1)+'_a'))+getattr(self,'b_t'+str(i+1)))

            else:
                setattr(self,'criticNet',tf.matmul(h1,getattr(self,'w'+str(i+1)))+getattr(self,'b'+str(i+1))) # doubtful as it bounds the actions to +1 to -1. Should we scale?
                setattr(self,'criticNet_t',tf.matmul(h1_t,getattr(self,'w_t'+str(i+1)))+getattr(self,'b_t'+str(i+1))) # doubtful as it bounds the actions to +1 to -1. Should we scale?

    def updateCritic(self,state,action,y_i):
        self.sess.run(self.optimizer,feed_dict={self.state_i:state,self.action_i:action,self.y_i:y_i})

    def updateTargetCritic(self):
        for i in range(int(len(self.theta_q)/2)):
            self.sess.run(getattr(self,'w_t'+str(i+1)).assign(self.tau*getattr(self,'w'+str(i+1))+getattr(self,'w_t'+str(i+1))*(1-self.tau)))  
            self.sess.run(getattr(self,'b_t'+str(i+1)).assign(self.tau*getattr(self,'b'+str(i+1))+getattr(self,'b_t'+str(i+1))*(1-self.tau)))
            if i == 1:
                self.sess.run(getattr(self,'w_t'+str(i+1)+'_a').assign(self.tau*getattr(self,'w'+str(i+1)+'_a')+getattr(self,'w_t'+str(i+1)+'_a')*(1-self.tau)))   
            
    def getdq_da(self,state,action):
        return self.sess.run(self.gradients,feed_dict={self.state_i:state,self.action_i:action})

    def getQValue(self,state,action):
        return self.sess.run(self.criticNet,feed_dict={self.state_i:state,self.action_i:action})

    def getTargetQValue(self,state,action):
        return self.sess.run(self.criticNet_t,feed_dict={self.state_i:state,self.action_i:action})

class OUNoise(object):
    """ docstring for OUNoise """
    def __init__(self,actionSize,mu=0, theta=0.15, sigma=0.2):
        self.actionSize = actionSize
        self.mu         = mu
        self.theta      = theta
        self.sigma      = sigma
        self.state      = np.ones(self.actionSize) * self.mu
        self.batchSize  = 64
        self.reset()

    def reset(self):
        self.state = np.ones(self.actionSize) * self.mu

    def generate(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class DDPG(object):
    """docstring for DDPG"""
    def __init__(self,environment):
        self.env = gym.make(str(environment))
        self.maxlenReplay = 1000000
        self.replay = deque(maxlen=self.maxlenReplay)
        self.batchSize = 64
        self.gamma = 0.99
        self.actionSpace = self.env.action_space.high.shape[0] 
        self.stateSpace = self.env.observation_space.high.shape[0]
        self.noise = OUNoise(self.env.action_space.high.shape[0])
        self.episodes = 10000

    def ddpg(self):
        actor = Actor(self.env.observation_space.high.shape[0],self.env.action_space.high.shape[0]) # Actor target already initialized within this
        critic = Critic(self.env.observation_space.high.shape[0],self.env.action_space.high.shape[0]) # Critic target already initialized within this
        count = 0
        samplable = False
        while count < self.episodes:
            self.noise.reset()
            noise = self.noise.generate()
            state_i = self.env.reset()
            state_i = np.reshape(state_i, [1, self.stateSpace])
            done = False
            while not done:
                action_i = actor.getAction(state_i) + noise
                [state_i2, reward_i, done, garbage] = self.env.step(action_i)
                if len(self.replay) >= self.maxlenReplay:
                    self.replay.popleft()
                self.replay.append([state_i,action_i,reward_i,state_i2,done])
                state_i = np.copy(state_i2)
                if not samplable:
                    if len(self.replay) >= self.batchSize:
                        samplable = True
                else:
                    batch = random.sample(self.replay,self.batchSize)
                    stateBatch_1 = [experience[0] for experience in batch]
                    stateBatch_1 = np.reshape(stateBatch_1,[len(stateBatch_1),self.stateSpace])
                    stateBatch_2 = [experience[3] for experience in batch]
                    stateBatch_2 = np.reshape(stateBatch_2,[len(stateBatch_2),self.stateSpace])
                    actionBatch_1 = [experience[1] for experience in batch]
                    actionBatch_1 = np.reshape(actionBatch_1,[len(actionBatch_1),self.actionSpace])
                    rewardBatch = [experience[2] for experience in batch]
                    rewardBatch = np.reshape(rewardBatch,[len(rewardBatch),1])
                    doneBatch = [experience[-1] for experience in batch]
                    actionBatch_2 = actor.getTargetAction(stateBatch_2)
                    actionBatch_2 = np.reshape(actionBatch_2,[len(actionBatch_2),self.actionSpace])
                    actionActual = actor.getAction(stateBatch_1)
                    actionActual = np.reshape(actionActual,[len(actionActual),self.actionSpace])
                    qValue_2 = critic.getTargetQValue(stateBatch_2,actionBatch_2)
                    yBatch[not doneBatch] = rewardBatch + self.gamma*qValue_2
                    #yBatch[doneBatch] = rewardBatch
                    #pdb.set_trace()
                    qValue_1 = critic.getQValue(stateBatch_1,actionBatch_1)
                    critic.updateCritic(stateBatch_1,actionBatch_1,yBatch)
                    actor.updateActor(stateBatch_1,critic.getdq_da(stateBatch_1,actionActual)[0])
                    critic.updateTargetCritic()
                    actor.updateTargetActor()                            
            count += 1

if __name__ == '__main__':
    environment = 'Walker2d-v2'
    agent = DDPG(environment)
    agent.ddpg()


# state_1_batch = []
# state_2_batch = []
# action_1_batch = []
# reward_batch = []
# for experience in batch:
    # state_1_batch.append(experience[0])
    # state_2_batch.append(experience[-1])
    # action_1_batch.append(experience[1])
    # reward_batch.append(experience[2])
