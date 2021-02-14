import gym
from gym.utils import seeding
from gym import spaces
import numpy as np
import random
import time
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import make_atari
import tensorflow as tf
from stable_baselines.common.callbacks import BaseCallback

from collections import OrderedDict 
#env = gym.make('Acrobot-v1')
# =============================================================================
# 
# import os
# import tensorflow as tf
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# =============================================================================


#del model # remove to demonstrate saving and loading

#model = DQN.load("deepq_Acrobot")

#obs = env.reset()
# Used to get all weight values that are nested within an ordered dict.
# =============================================================================
# def ConvertWeightValuesTo1DVector(weightsOrderedDict):
#     weightVector = np.array([])
#     for key, value in weightsOrderedDict.items():
#         keyShape = weightsOrderedDict[key].shape
#         if not keyShape == ():
#             for rowIndex in range(keyShape[0]):                
#                 if weightsOrderedDict[key][rowIndex].size > 1:
#                     for rowElement in weightsOrderedDict[key][rowIndex]: # 'value' can probably be used as iteration object instead
#                         weightVector = np.append(weightVector, rowElement)
#                 else:
#                     weightVector = np.append(weightVector, weightsOrderedDict[key][rowIndex])
#         else:
#             weightVector = np.append(weightVector, value)
#     return weightVector
# =============================================================================


# Used to get all weight values that are nested within an ordered dict.
def ConvertWeightValuesTo1DVector(weightsOrderedDict):
    original_shape = {}
    weightVector = []
    for key in weightsOrderedDict:
        original_shape[key]=weightsOrderedDict[key].shape
        if weightsOrderedDict[key].shape == ():
            weightVector.append(weightsOrderedDict[key])
        elif len(weightsOrderedDict[key].shape) == 2:
            x = weightsOrderedDict[key].shape[0]
            y = weightsOrderedDict[key].shape[1]
            [weightVector.append(i) for i in weightsOrderedDict[key].reshape((x*y))]
        elif len(weightsOrderedDict[key].shape) == 4:
            x = weightsOrderedDict[key].shape[0]
            y = weightsOrderedDict[key].shape[1]
            z = weightsOrderedDict[key].shape[2]
            w = weightsOrderedDict[key].shape[3]
            [weightVector.append(i) for i in weightsOrderedDict[key].reshape((x*y*z*w))]
        elif len(weightsOrderedDict[key].shape) == 1:
            [weightVector.append(i) for i in weightsOrderedDict[key]]
    weightVector = np.array(weightVector)
    return weightVector, original_shape

def Convert1DWeightVectorToModelParameters(weightVector, shapes):
    new_dick = {}
    counter = 0 
    for i in shapes:
        if shapes[i] == ():
            new_dick[i] = weightVector[counter]
            counter = counter + 1
        elif len(shapes[i]) == 2:
            x = shapes[i][0]
            y = shapes[i][1]
            temp_array = weightVector[counter:counter+x*y]
            new_dick[i] = temp_array.reshape((x,y))
            counter = counter+x*y
        elif len(shapes[i]) == 4:
            x = shapes[i][0]
            y = shapes[i][1]
            z = shapes[i][2]
            w = shapes[i][3]
            temp_array = weightVector[counter:counter+(x*y*z*w)]
            new_dick[i] = temp_array.reshape((x,y,z,w))
            counter = counter+(x*y*z*w)
        elif len(shapes[i]) == 1:
            x = shapes[i][0]
            temp_array = weightVector[counter:counter+x]
            new_dick[i] = temp_array
            counter = counter+x
    
    return OrderedDict(new_dick)



class TestCallback(BaseCallback):
    def _on_step(self):
        #if self.timestep == 1:
        shit = self.locals
        #print(shit['episode_rewards'])
        
    #def _on_training_start(self):
       # assert 'total_timesteps' not in self
                    
# =============================================================================
# def GetWeightBatch(outBatch):
#     skatoStrings = []
#     for key in weightsOrderedDict:
#         skatoStrings = key.split("/")
#         gamotoString = skatostrings[-1].split(":")
# 
# =============================================================================
"""
Compute the inner system's current state, according to the inner environment's total rewards at this point.
"""
def GetCurrentInnerSystemState(reward, initialWithLastAgentDifference):
    return [reward, initialWithLastAgentDifference.tolist()] # Used for when inner model learns
    #return [0, [0]]
    #return [reward, [reward]] # Used when inner model does not learn


"""
Inner system within the meta-learning process, implemented with a Q-Learning algorithm.
The inner system consists of the inner (input) environment, as well as the agent,
and the rest of parameters that defined the interaction between the two.
These include the outer reward that this system shall send to the outer system (meta-learner)
as well as the current state communicated to the meta-learner. This state is a vector
composed by the inner agent's current (total) reward, as well as the inner system's state (?).    
"""
class InnerSystemDQN(gym.Env):
    def __init__(self, innerEnv, outBatch = 0, noBatches = True, seed = 0, agent = None):
        
        #self.innerEnv = innerEnv() # Inner environment.
        self.innerEnv = gym.make(innerEnv)
        #self.innerEnv = make_atari(innerEnv)
        #self.innerEnv = self.innerEnv.env
        self.epsilon = 0.1 # Probability of selecting a random action.
        self.gamma = 0.99 # Discount factor.
        self.alpha = 0.5 # Learning rate.
        self.episodeCount = 1 # Number of episodes.

        self.innerSystemReward = 0
        self.temp_state = []
        self.innerSystemState = []
        
        #directory = "DQN"
        self.logDir = "./ppo2_innersys1/" #+ directory
        
        #self.model = DQN(MlpPolicy, self.innerEnv, policy_kwargs=dict(dueling=False), verbose=0, 
                         #double_q=False, learning_starts=1, target_network_update_freq=1, seed = 0)
        self.model = DQN(MlpPolicy, self.innerEnv, verbose=0, seed = 0)                 

        # Keep track of rewards and steps in each episode.
        #self.episodeRewards = np.zeros(self.episodeCount)
        #self.episodeSteps = np.zeros(self.episodeCount)
        self.printShit = False
# =============================================================================
# ----------        OLD         -------------------
#         Might be moved to another class
#         # Currently, the state vector is actually the reward.   
#         self.low_state = np.array([-np.inf])
#         self.high_state = np.array([np.inf])
# =============================================================================
        modelWeights = self.model.get_parameters()
        self.modelWeightVector, self.originalShapes = ConvertWeightValuesTo1DVector(modelWeights)
        self.initialModelParameters = modelWeights
        self.model.save("deepq")
        if noBatches == True:
            self.numberOfBatches = 1 # Unnecessary I think
            self.startBatchVector = 0
            self.endBatchVector = len(self.modelWeightVector)
        else:
            self.numberOfBatches = 100
            self.outBatch = outBatch
            self.outBatch = self.outBatch * int((len(self.modelWeightVector) - 1) / self.numberOfBatches)
            self.startBatchVector = self.outBatch + 1
            self.endBatchVector = self.outBatch + int((len(self.modelWeightVector) - 1) / self.numberOfBatches)
        
        
        # Crease observation space (which has the same form as the weight vector of the trained DQN agent).
        # Check for more details in action space creation below (is same as observation space creation).
        random.seed(0)
        
        self.weightIndices = random.sample(range(len(self.modelWeightVector)), 
                                self.endBatchVector - self.startBatchVector)
        
        #### TO BE USED ONLY IF REWARD IS PART OF THE OBSERVATION SPACE ###
        # self.low_state = np.array([-np.inf])
        # self.high_state = np.array([np.inf])
        self.low_state = np.array([])
        self.high_state = np.array([])
        for weight in self.modelWeightVector[self.startBatchVector:self.endBatchVector]:
            self.low_state = np.append(self.low_state, -np.inf)
            self.high_state = np.append(self.high_state, np.inf)  
        
        
            # Instead of normalizing
            #self.low_state = np.append(self.low_state, -1)
            #self.high_state = np.append(self.high_state, 1) 
        # TAKE NOTE, BECAUSE THIS IS SERIOUS:
        # WHEN ACCESSING/FEEDING PARAMETERS OF/TO THE MODEL,
        # THE FIRST ONE IS "EPSILON". WE HAVE TO CHANGE THE BOUNDS
        # OF THIS ELEMENT WITHIN THE ACTION/SPACE VECTOR TO BE
        # WITHIN 0-0.99.
        #self.low_state[0] = 0.999
        #self.high_state[0] = 0.999
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        self.observation_space = self.innerEnv.observation_space
        # This is not general! Has to be generalized. Shall (hopefully) work for this experiment though.
        # An action in this environment means to generate a new DQN agent. Therefore, the
        # action space has the same form of the weight vector of a DQN agent
        # (i.e. a -np.inf (low) and an np.inf (high))
        self.low_action = []
        self.high_action = np.array([])
        #spaceTuples.append(spaces.Discrete(totalSpace))
        for weight in self.modelWeightVector[self.startBatchVector:self.endBatchVector]:
            self.low_action = np.append(self.low_action, -np.inf)
            self.high_action = np.append(self.high_action, np.inf)   
            # Instead of normalizing
            #self.low_action = np.append(self.low_action, -1)
            #self.high_action = np.append(self.high_action, 1) 
        # TAKE NOTE, BECAUSE THIS IS SERIOUS:
        # WHEN ACCESSING/FEEDING PARAMETERS OF/TO THE MODEL,
        # THE FIRST ONE IS "EPSILON". WE HAVE TO CHANGE THE BOUNDS
        # OF THIS ELEMENT WITHIN THE ACTION/SPACE VECTOR TO BE
        # WITHIN 0-0.99.
        self.low_action[0] = 0.999
        self.high_action[0] = 0.999
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        
        self.mounoseed = seed
        self.innerCounter = 0
        
        ###### START DANGER ######
        #self.innerEnv._max_episode_steps = 200
        ###### END DANGER ######
        
        self.oldReward = 0
        self.rewardDiff = 0
        self.lastAction = None
        self.totalEps = 0        

    """
    Reset the whole inner system. This includes resetting the inner environment,
    as well as the inner agent and the parameters defined for this system.
    """
    def reset(self, episodeCount = 10, epsilon = 0.1, gamma = 0.99, alpha = 0.5):
        
        # Reset inner environment.
        self.innerState = self.innerEnv.reset()
        #self.innerEnv.seed(self.mounoseed)
        #self.innerEnv._max_episode_steps = 500
        
        # Reset inner system parameters.
        self.epsilon = epsilon # Probability of selecting a random action.
        self.gamma = gamma # Discount factor.
        self.alpha = alpha # Learning rate.
        self.episodeCount = episodeCount # Number of episodes.
        #self.windowSize = 10 # Number of steps to use for smoothing the plots...
        
        self.innerSystemReward = 0
        self.innerSystemState = []
        
# =============================================================================
#         if self.innerCounter % 10 == 0:
#             print(self.innerCounter)
#             print(self.innerSystemReward)
# =============================================================================
            #directory = "DQN" + str(self.innerCounter)
            #self.logDir = "./ppo2_innersys/" + directory
        
        
        # Reset agent (for now, it's simply a Q-table).    
        #self.model = DQN(MlpPolicy, self.innerEnv, verbose=0, learning_starts=10, target_network_update_freq=10, seed = 0)       
        
        # self.modelWeightVector will always exist at this point I think
        #completeAction = self.modelWeightVector
        #newModelWeights = Convert1DWeightVectorToModelParameters(completeAction, self.originalShapes)
        #self.model.setup_model()
        self.model.load_parameters(self.initialModelParameters, exact_match = True)
        
        
        #for state in range(0,self.innerEnv.observation_space.n):
            #self.Q[state] = np.zeros(self.innerEnv.action_space.n)
        modelWeights = self.model.get_parameters()
        self.modelWeightVector, self.originalShapes = ConvertWeightValuesTo1DVector(modelWeights)
    
        # Keep track of rewards and steps in each episode.
        #self.episodeRewards = np.zeros(self.episodeCount)
        #self.episodeSteps = np.zeros(self.episodeCount)
        
    
        # Reset the rest of the inner system.
        self.innerSystemReward = 0 
        #self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.modelWeightVector[self.startBatchVector:self.endBatchVector])
        self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.innerState)
        #self.innerCounter = self.innerCounter + 1
        
        #self.lastAction = self.modelWeightVector
        self.totalEps = 0
        return self.innerSystemState
    
    """
    This function is called every time the meta-learner performs an action in his own, outer system.
    At this point, this method shall make a whole round of interaction between the inner agent agent 
    and the inner environment. In other words, one step of the outer agent means several steps for the
    inner agent.
    Currently, it uses uses the Q-learning algorithm to returns the optimal action-value function Q, 
    a dictionary mapping state -> action values, following an epsilon-freedy policy.
    """
    def step(self, action):
        done = False
        info = {"done: " : done}
        shit = self.innerEnv.reset()
        #self.innerEnv.seed(self.mounoseed)

        #if self.innerCounter % 1 == 0:
            #print(self.innerCounter)          
        #if self.innerCounter % 8 == 0:
            #print("This is it!")
        #action = (np.tanh(action) + 1) / 2 * (self.action_space.high - self.action_space.low) + self.action_space.low
        #action =  ((action - min(action)) / (max(action) - min(action)) - 0.5) * 2
        
        for i in range(self.endBatchVector - self.startBatchVector):
            self.modelWeightVector[self.weightIndices[i]] = action[i]
        #self.modelWeightVector[self.startBatchVector:self.endBatchVector] = action
        #print(action)
        completeAction = self.modelWeightVector
        
        newModelWeights = Convert1DWeightVectorToModelParameters(completeAction, self.originalShapes)
        #self.model.setup_model()
        
        self.model.load_parameters(newModelWeights, exact_match = True)

        
        #self.model.learn(total_timesteps=10, callback=TestCallback())
        #start_time = time.time()
        #self.model.save("deepq")        
        #print("Time for saving model: %s seconds" % (time.time() - start_time))
        
        self.innerSystemReward = 0
        
        eps = 10
        temp_reward = np.zeros(eps)
        
        for ep in range(eps):
            done = False
            self.temp_state = []
            innerState = self.innerEnv.reset()
            self.temp_state.append(innerState)
            count = 0
            while done == False:

                innerAction, _states = self.model.predict(innerState)
                #print(innerAction)
                innerState, innerReward, done, info = self.innerEnv.step(innerAction)
                
                #if done == True:
                #    print("Shit this is shit shit shit")
                    #print(count)
                # Below is the reward array that keeps track of rewards coming until _done_ flag is raised.
                # This is due to the nature of the environment.
                temp_reward[ep] = temp_reward[ep] + innerReward
                self.temp_state.append(innerState)
                count = count + 1
                #self.innerEnv.render()
                
                # =============================================================================
                #if count == 200:
                #    done = True
                    #print("Well shit")
                # =============================================================================
            #print(ep)
            #print(temp_reward[ep])
        #print("Outer episode: {0}, Average Reward: {1}".format(self.innerCounter, self.innerSystemReward / 50.0))
        #modelWeights = self.model.get_parameters() # Not necessary if no DQN training takes place
        outerEps = 1
        self.innerSystemReward = sum(temp_reward) / (eps)
        
        
        ##### START TESTING A GOOD AGENT THOROUGHLY #####
# =============================================================================
#         shitTestEps = 10
#         if self.innerSystemReward > 200:
#             temp_reward = np.zeros(shitTestEps)
#             for ep in range(shitTestEps):
#                 done = False
#                 #self.temp_state = []
#                 innerState1 = self.innerEnv.reset()
#                 #self.temp_state.append(innerState1)
#                 count = 0
#                 while done == False:
#     
#                     innerAction, _states = self.model.predict(innerState1)
#                     #print(innerAction)
#                     innerState1, innerReward, done, info = self.innerEnv.step(innerAction)
#                     
#                     #if done == True:
#                     #    print("Shit this is shit shit shit")
#                         #print(count)
#                     # Below is the reward array that keeps track of rewards coming until _done_ flag is raised.
#                     # This is due to the nature of the environment.
#                     temp_reward[ep] = temp_reward[ep] + innerReward
#                     #self.temp_state1.append(innerState1)
#                     #count = count + 1
#                     #self.innerEnv.render()
#             print("OK LETS SEE IF ITS GOOD:")
#             print("Average Reward: {0}".format(sum(temp_reward) / (shitTestEps)))
#             done = True
# =============================================================================
        ##### END TESTING A GOOD AGENT THOROUGHLY #####
            
            
        self.temp_state = np.sum(self.temp_state,0) / count
        self.rewardDiff = self.innerSystemReward - self.oldReward
        self.oldReward = self.innerSystemReward
        
        
        if self.printShit == True:
            print("Reward diff: {0}".format(self.rewardDiff))
            #print("Old Reward: {0}".format(self.oldReward))
            print("Average score: {0}".format(self.innerSystemReward))
            
        # Not necessary if no DQN training takes place
        #self.modelWeightVector, self.originalShapes = ConvertWeightValuesTo1DVector(modelWeights)

        if self.lastAction is not None:
            self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, action - self.lastAction)
            #self.innerSystemState = GetCurrentInnerSystemState(innerState, action)
        else:
            self.innerSystemState = GetCurrentInnerSystemState(innerState, action)
        self.innerCounter = self.innerCounter + 1
        #PlotPerformanceFigures(1, self.episodeRewards, self.episodeSteps)
        
        # I think this should be used only when the batch weight vector strategy is not used,
        # and when the state vector is defined as the difference between last action and current
        self.lastAction = action
        self.totalEps = self.totalEps + 1
# =============================================================================
#         if self.totalEps < outerEps:
#             done = False
#         else:
#             doen = True
# =============================================================================
        
        return (self.innerSystemState, self.innerSystemReward, done, info)


# =============================================================================
#     def GetCurrentAgent(self):
#         return self.Q
#     
#     def SetCurrentAgent(self, agent):
#         self.Q = agent
# =============================================================================
        
def seed(self, out_seed = None):
    np_random, self.mounoseed = self.innerEnv.seed(out_seed)
    return [self.mounoseed]
        
"""  
Close inner environment (for practical reasons).
"""   
def CloseInnerEnvironment(self):
    self.innerEnv.close()