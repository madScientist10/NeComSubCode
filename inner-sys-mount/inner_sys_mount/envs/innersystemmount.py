import gym
import numpy as np
from tiles3 import tiles, IHT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D

from gym import spaces
from gym.utils import seeding

#env = gym.make("MountainCar-v0")


"""
Compute the inner system's current state, according to the inner environment's total rewards at this point.
"""
def GetCurrentInnerSystemState(reward, lastAgentVersion):
    #concatStateVec = reward + lastAgentVersion

    return [reward, lastAgentVersion]


class Tilecoder:

    def __init__(self, innerEnv, numTilings):
        
       

        # Store a few basic variables from environment
        # and properties of tiler
        self.maxVal = innerEnv.observation_space.high
        self.minVal = innerEnv.observation_space.low
        self.maxSize = 1024
        self.numTilings = 5
        self.stepSize = 0.5/numTilings
#        self.alpha = alpha
        self.actions = innerEnv.action_space.n
        self.n = self.maxSize * self.actions
        
     


    # This is feature the vector that represents the gradient
    # of the approximate value function with respect to
    # the weight vector.
    # For the linear case, it has a very simple form.
    # Its components are 1 at the position where the
    # corresponding feature is found, and zero everywhere else.
    def getFeatureVector(self, numTiles, features, action):
        featureVector = np.zeros(self.n)
        for feature in features:
            index = int(feature + (numTiles*action))
            featureVector[index] = 1
        return featureVector


    # Get the corresponding Q table as a function of theta 
    # (i.e. weights), which, for the linear case,
    # in essence, is the feature that represents the
    # current state.
    def getQ(self, numTiles, features, theta):
        Q = np.zeros(self.actions)
        # For each action,
        for action in range(self.actions):
            val = 0
            # For each feature,
            for feature in features:
                # Get the value that corresponds to Q[action]
                # as a function of theta
                index = int(feature + (numTiles*action))
                val += theta[index]
            Q[action] = val
        return Q
    
    
    def policy_fn(self, Q, epsilon):
    
        """
        Epsilon-greedy policy.
        
        Input:
            q: Action-value function approximation.
            epsilon: Probability of selecting a random action.
            nA: Number of actions in the environment.
            
        Output:
            Probabilities for each action.
        """
        A = np.ones(self.actions, dtype=float) * epsilon / self.actions
        best_action = np.argmax(Q)
        A[best_action] += (1.0 - epsilon)
        return A


class InnerSystemMount(gym.Env):

    def __init__(self, innerEnv):
        
        self.innerEnv = innerEnv() # Inner environment.
        self.innerSystemReward = 0
        self.innerSystemState = []
        self.episodeCount = 1000

        self.numTilings = 32
        self.alpha = 0.5 / self.numTilings
        self.epsilon = 0.1
        self.gamma = 0.99
        
        self.tile = Tilecoder(self.innerEnv, 32)
        
        # Initialize weights
        self.weights = np.zeros((self.episodeCount, self.tile.n))
        self.theta = np.zeros(self.tile.n)
        self.iht = IHT(self.tile.maxSize)
        
        
        
        # Might be moved to another class
        # Currently, the state vector is actually the reward.
        self.low_state = np.array([-np.inf])
        self.high_state = np.array([np.inf])
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        
        # This is not general! Has to be generalized. Shall (hopefully) work for this experiment though.
        #totalSpace = len(self.Q) * len(self.Q[0])
        self.low_action = []
        self.high_action = np.array([])
        #spaceTuples.append(spaces.Discrete(totalSpace))

        for entryLength in self.theta:
            self.low_action = np.append(self.low_action, -np.inf)
            self.high_action = np.append(self.high_action, np.inf)
        
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        
        # Keep track of rewards and steps in each episode.
        self.episodeRewards = np.zeros(self.episodeCount)
        self.episodeSteps = np.zeros(self.episodeCount)
        
    
        # Reset the rest of the inner system.
        #self.lastaction = None
        self.innerSystemReward = 0        
        self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.theta.tolist())
        
        self.seed = self.seed(0)
        
        
        
    def reset(self, episodeCount = 1000, epsilon = 0.1, gamma = 0.99, alpha = 0.5):
                            
        self.innerEnv.reset()
        # Number of episodes
        self.episodeCount = episodeCount
        self.numTilings = 32
        self.alpha = alpha / self.numTilings
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.tile = Tilecoder(self.innerEnv, 32)
        
        # Initialize weights
        self.weights = np.zeros((self.episodeCount, self.tile.n))
        self.theta = np.zeros(self.tile.n)
        self.iht = IHT(self.tile.maxSize)
        
        # Keep track of rewards and steps in each episode.
        self.episodeRewards = np.zeros(self.episodeCount)
        self.episodeSteps = np.zeros(self.episodeCount)
        
    
        # Reset the rest of the inner system.
        #self.lastaction = None
        self.innerSystemReward = 0        
        self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.theta.tolist())
        
        return self.innerSystemState
    
    
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

        
        
    def step(self, action):
        done = False
        info = {"done: " : done}
        self.innerSystemReward = 0
        self.episodeRewards = np.zeros(self.episodeCount)

        action = self.theta
        #action = action
        # Number of steps to use for smoothing the plots...
        windowSize = 60
    
        # Initialize state vector, as well as
        # tables for rewards and steps.
        values = np.zeros(len(self.tile.maxVal))        
        episodeRewards = np.zeros(self.episodeCount)
        episodeSteps = np.zeros(self.episodeCount, dtype=int) 
    
        # Ok, begin an episode.
        for episodeNum in range(self.episodeCount):
            
            #if episodeNum == 2:
            #    epsilon = 0
            
            # Initialize innerReward and innerState.
            G = 0
            innerState = self.innerEnv.reset()
            #action = 0
            while True:
                #env.render()
                
                # Update step count.
                episodeSteps[episodeNum - 1] += 1

                # Get featurized innerState values.
                stateValues = [1*innerState[0]/(0.5+1.2), 1*innerState[1]/(0.07+0.07)]
                
                # Get tile indices.
                tileIndices = tiles(self.iht, 32, stateValues)
                
                # Get Q.
                Q = self.tile.getQ(self.tile.maxSize, tileIndices, self.theta)
                
                
                # Compute next innerAction, innerState, innerReward, etc.
                # Compute initial action using defined policy
                action_probs = self.tile.policy_fn(Q, self.epsilon)
                bestInnerAction = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                         
                #bestInnerAction = np.argmax(Q)
                nextInnerState, innerReward, done, info = self.innerEnv.step(bestInnerAction)
                G += innerReward
                #episodeRewards[episodeNum] = G
                delta = innerReward - Q[bestInnerAction]
                
                # Store total rewards at this point so as to be ready to use it when asked by the outer
                # system.
                self.innerSystemReward += innerReward
                
                # If goal is reached...
                if done == True:
                    
                    # ...update weight vector.
                    self.theta += np.multiply((self.alpha*delta), self.tile.getFeatureVector(self.tile.maxSize, tileIndices, bestInnerAction))
                    self.weights[episodeNum - 1] = self.theta
                    # Also make a sum of this episode's rewards.
                    episodeRewards[episodeNum - 1] = G
                    
                    #PlotPerformanceFigures(1, self.episodeRewards, self.episodeSteps)
                    return (self.innerSystemState, self.innerSystemReward, done, info)
                    
                    #averageStepValues[episodeNum - 1, run - 1, a - 1] = episodeSteps[episodeNum - 1]                    
                    # Print current episode's innerReward once in a while...
#                        if episodeNum % 100 == 0:
#                            print('Episode {} average innerReward = {}'.format(episodeNum, episodeRewards[episodeNum - 1]))
                    break # ...and start new episode.
                
                # Otherwise, if goal is not reached yet, then compute next featurized state values
                stateValues2 = [1*nextInnerState[0]/(0.5+1.2), 1*nextInnerState[1]/(0.07+0.07)]
                
                # Now, get Q values for next state values
                newTileIndices = tiles(self.iht, 32, stateValues2)
                Q = self.tile.getQ(self.tile.maxSize, newTileIndices, self.theta)
                
                # Compute initial action using defined policy
                #action_probs = tile.policy_fn(Q, epsilon)
                #bestInnerAction = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                                             
                # Compute (gamma * max{Q(s',a)})...
                delta += self.gamma*np.max(Q)
                
                # Now it's necessary to update weights.
                self.theta += np.multiply((self.alpha*delta), self.tile.getFeatureVector(self.tile.maxSize, tileIndices, bestInnerAction))
                #weights[episodeNum - 1, episodeSteps[episodeNum - 1]] = theta

                # Ok, update state now and repeat.
                innerState = nextInnerState
                self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.theta.tolist())
                
            return (self.innerSystemState, self.innerSystemReward, done, info)
            
                
    
#    plt.figure(1)
#    finalAverage = np.zeros((self.episodeCount, alphaValues))
#
#    for a in range(1, alphaValues + 1):
#        for ep in range(1,self.episodeCount+1):
#            for i in range(1, runs + 1):
#                finalAverage[ep - 1, a - 1] += averageStepValues[ep - 1, i - 1, a - 1] / runs
#        plt.plot(finalAverage[:, a - 1], linewidth = 1.0)
#
##
        
        
#    episodeSteps_Smoothed = np.zeros(episodeNum)
#    weights = np.repeat(1.0, windowSize) / windowSize
#    for wght in range(episodeNum - 1):
#        episodeSteps_Smoothed = np.convolve(finalAverage[:,0], weights, 'valid')
#         
#    plt.plot(episodeSteps_Smoothed[:], linewidth = 1.0)
    # create rainbow colors
  #  freq = 0.1 #255/len(weights[1:]);
#    for i in range(len(weights[1,:])):
#        red = (np.sin(freq*i + 2) * (127) + 128)/255;
#        green = (np.sin(freq*i + 0) * (127) + 128)/255;
#        blue = (np.sin(freq*i + 4) * (127) + 128)/255;
#        plt.plot(weights[:,i], color = (red, green, blue))
def PlotPerformanceFigures(weights):
    plt.figure(1)
    average = np.zeros(10000)
    for j in range(1, 10000):    
        for i in range(1, 12288):
            if weights[j,i] == 0:
                continue
            average[j] += weights[j, i]
        average[j] = average[j]/12288
    plt.plot(average)        
    plt.show()
        
#    print("Printing plots...")
##    plt.figure(1)
#    weights = np.repeat(1.0, windowSize) / windowSize
#    episodeRewards_Smoothed = np.convolve(episodeRewards, weights, 'valid')
#    #episodeRewards_Smoothed = pd.Series(episodeRewards).rolling(windowSize, min_periods=windowSize).mean()
##    plt.plot(episodeRewards_Smoothed, linewidth = 1.0)
#    #plt.plot(episodeRewards, linewidth = 1.0)
#    plt.figure(2)
##    for a in range(1,alphaValues):
##        episodeSteps_Smoothed = np.convolve(averageStepValues[:a], weights, 'valid')
#    
#    #plt.plot(episodeSteps, linewidth = 1.0)
#    
#    #line.set_antialiased(False)
#    plt.show()

#env.close()







