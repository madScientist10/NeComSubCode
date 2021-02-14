import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class MultiInnerInterface(gym.Env):
    
    def __init__(self, innerVec):
        innerEnvVecName = innerVec[0]
        innerEnvVecId = innerVec[1]
        innerEnvVec = innerVec[2]
        
        self.innerEnvVecName = []
        for EnvVecName in innerEnvVecName:    
            self.innerEnvVecName.append(EnvVecName)
            
        self.innerEnvVecId = []
        for EnvVecId in innerEnvVecId:
            self.innerEnvVecId.append(EnvVecId)
        
        self.innerEnvVecString = []
        for i in range(len(self.innerEnvVecId)):
            self.innerEnvVecString.append(self.innerEnvVecName[i] + ":" + self.innerEnvVecId[i])
        
        self.innerEnvVec = []
        for EnvVec in innerEnvVec:
            self.innerEnvVec.append(EnvVec)
        
    
        
        """
        De-register environment from previous run (so as not to restart). Quick fix.
        Might be moved elsewhere (e.g. as innerSystem method)
        """
        env_dict = gym.envs.registration.registry.env_specs.copy()
        for env in env_dict:
            for innerEnv in self.innerEnvVecId:
                if innerEnv in env:
                     print('Remove {} from registry'.format(env))
                     del gym.envs.registration.registry.env_specs[env]
        
        self.readyEnvs = []
        for envInd in range(len(self.innerEnvVecString)):
            self.readyEnvs.append(gym.make(self.innerEnvVecString[envInd], innerEnv = self.innerEnvVec[envInd])) # Environment.


        #self.observation_space = []
        #self.action_space = []
        self.low_state = np.zeros(len(self.readyEnvs))
        self.high_state =np.zeros(len(self.readyEnvs))
        self.spaceTuples_low = np.zeros(len(self.readyEnvs))
        self.spaceTuples_high = np.zeros(len(self.readyEnvs))
        for ind in range(len(self.readyEnvs)):            
            np.insert(self.low_state,ind, self.readyEnvs[ind].low_state)
            np.insert(self.high_state,ind, self.readyEnvs[ind].high_state)
            np.insert(self.spaceTuples_low, ind, self.readyEnvs[ind].spaceTuples_low)
            np.insert(self.spaceTuples_high, ind, self.readyEnvs[ind].spaceTuples_high)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=self.spaceTuples_low, high=self.spaceTuples_high)
        
    
        self.readyEnvRewards = []
        self.readyEnvSeed = 0
    
        
    def reset(self):
         for readyEnv in self.readyEnvs:
            readyEnv.reset()
            
    def step(self, action):
         for envInd in range(len(self.readyEnvs)):
            self.readyEnvRewards[envInd] = self.readyEnvs[envInd].step(action[envInd])
    
         return self.readyEnvRewards
    
    def seed(self):
        for readyEnv in self.readyEnvs:
            readyEnv.seed(self.readyEnvSeed)