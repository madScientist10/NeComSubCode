import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines.common.atari_wrappers import make_atari



class MultiInnerInterface(gym.Env):
    
    def __init__(self, env_args = None):
        
        if env_args is not None:
            innerVec = env_args['innerVec']
            outBatch = env_args['outBatch']
            noBatches = env_args['noBatches']
            differentAlgo = env_args['differentAlgo']
        else:
            print('No args given for multiinterface shit')
        
        
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
                     print('Removing {} from registry'.format(env))
                     del gym.envs.registration.registry.env_specs[env]
        
        self.readyEnvs = []
        for envInd in range(len(self.innerEnvVecString)):
            self.readyEnvs.append(gym.make(self.innerEnvVecString[envInd], 
                                            innerEnv = self.innerEnvVec[envInd], 
                                            outBatch = outBatch,
                                            noBatches = noBatches)) # Environment.
# =============================================================================
#             self.readyEnvs.append(make_atari(self.innerEnvVecString[envInd], 
#                                            innerEnv = self.innerEnvVec[envInd], 
#                                            outBatch = outBatch)) # Environment.
# =============================================================================
            # For CartPole-v1
            #self.readyEnvs[envInd] = self.readyEnvs[envInd].unwrapped
            #self.readyEnvs[envInd]._max_episode_steps = 200


        self.differentAlgo = differentAlgo

        # Create observation space
        self.low_state = np.array([])
        self.high_state =np.array([])
        
        
        if self.differentAlgo == True:
            # This is for the case where different types of algorithms are to be used
            for env in self.readyEnvs:
                for element in range(len(env.low_action)):
                    self.low_state = np.insert(self.low_state,element, env.low_state)
                    self.high_state = np.insert(self.high_state,element, env.high_state)  
            # Add reward element in observation vector
            #self.low_state = np.insert(self.low_state, 0, env.low_state)
            #self.high_state = np.insert(self.high_state,0, env.high_state)  
            
            # Make final observation space
            self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                                dtype=np.float32)
            self.observation_space = self.readyEnvs[0].observation_space
            # Create action space
            self.low_action = np.array([])
            self.high_action = np.array([])
            for ind in range(len(self.readyEnvs)):
                #self.low_state = np.insert(self.low_state,ind, self.readyEnvs[ind].low_state)
                #self.high_state = np.insert(self.high_state,ind, self.readyEnvs[ind].high_state)
                self.low_action = np.insert(self.low_action, ind, self.readyEnvs[ind].low_action)
                self.high_action = np.insert(self.high_action, ind, self.readyEnvs[ind].high_action)
            self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
                    
            
                
        else:
# =============================================================================
#             for element in range(len(self.readyEnvs[0].low_action)):
#                 self.low_state = np.insert(self.low_state,element, self.readyEnvs[0].low_state)
#                 self.high_state = np.insert(self.high_state,element, self.readyEnvs[0].high_state)
# =============================================================================
                   
            self.low_state = self.readyEnvs[0].low_state
            self.high_state = self.readyEnvs[0].high_state
            # Add reward element in observation vector
            #self.low_state = np.insert(self.low_state, 0, env.low_state)
            #self.high_state = np.insert(self.high_state,0, env.high_state)  
            
            # Make final observation space
            self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                                dtype=np.float32)
            self.observation_space = self.readyEnvs[0].observation_space
            # Create action space
            self.low_action = np.array([])
            self.high_action = np.array([])
           
            #self.low_state = np.insert(self.low_state,ind, self.readyEnvs[ind].low_state)
            #self.high_state = np.insert(self.high_state,ind, self.readyEnvs[ind].high_state)
            ind = 0
            self.low_action = np.insert(self.low_action, ind, self.readyEnvs[ind].low_action)
            self.high_action = np.insert(self.high_action, ind, self.readyEnvs[ind].high_action)
            self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        

        self.readyEnvRewards = np.zeros(len(self.readyEnvs))
        self.readyEnvSeed = 0
    
        self.innerStates = [] #np.zeros(len(self.readyEnvs))
        

        
    def reset(self):
        self.innerStates = [] # np.zeros(len(self.readyEnvs))
        self.innerVector = []
        for env in self.readyEnvs:
            self.innerVector = self.innerVector + env.reset()[1]
            #env.reset()
            #self.innerStates.append(readyEnv.reset())
        #self.innerStates = np.insert(self.innerStates, ind, [self.readyEnvs[ind].reset()])
        self.innerStates = self.innerVector
        

        return self.innerStates
            
           
    def step(self, action):
        done = np.zeros(len(self.readyEnvs), dtype=bool)
        info = {}
        for i in range(len(done)):
            done[i] = False
            info["done_" + str(i) + ":"] = done[i]
        if self.differentAlgo == True:   
            for envInd in range(len(self.readyEnvs)):
                env_action_start = 0
                env_action_end = 0
                for eachEnv in range(envInd):
                    env_action_start = env_action_start + len(self.readyEnvs[eachEnv].action_space.low)
                for eachEnv in range(envInd + 1):
                    env_action_end = env_action_end + len(self.readyEnvs[eachEnv].action_space.low)
                     
                self.innerStates[envInd], self.readyEnvRewards[envInd], done[envInd], info["done_" + str(envInd) + ":"] = self.readyEnvs[envInd].step(action[env_action_start:env_action_end])
        else:   
            for envInd in range(len(self.readyEnvs)):                 
                self.innerStates[envInd], self.readyEnvRewards[envInd], done[envInd], info["done_" + str(envInd) + ":"] = self.readyEnvs[envInd].step(action)

        self.innerVector = []
        for envInd in range(len(self.readyEnvs)):
            self.innerVector = self.innerVector + self.innerStates[envInd][1]
            #self.innerVector = self.innerVector + self.readyEnvs[envInd].reset()[1]
        return (self.innerVector, sum(self.readyEnvRewards), np.prod(done), info)
    
    def seed(self, seed = None):
        np_randoms = np.zeros(len(self.readyEnvs))
        seeds = np.zeros(len(self.readyEnvs))
        for envInd in range(len(self.readyEnvs)):
            seeds[envInd] = self.readyEnvs[envInd].seed(seed)
        return [seeds[0]]