import gym
import numpy as np
import pandas as pd
from EpsilonGreedyPolicy import epsilon_greedy_policy
from collections import defaultdict
import matplotlib.pyplot as plt
#from InnerEnv import CliffWalkingEnv
from gym.utils import seeding
from gym import spaces

#env = GridworldEnv()
# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
outdir = '/tmp/random-agent-results'

#env = wrappers.Monitor(env, directory=outdir, force=True)
#env.seed(0)

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


"""
Compute the inner system's current state, according to the inner environment's total rewards at this point.
"""
def GetCurrentInnerSystemState(self):
    return [self.innerSystemReward, self.Q]


"""
Inner system within the meta-learning process, implemented with a Q-Learning algorithm.
The inner system consists of the inner (input) environment, as well as the agent,
and the rest of parameters that defined the interaction between the two.
These include the outer reward that this system shall send to the outer system (meta-learner)
as well as the current state communicated to the meta-learner. This state is a vector
composed by the inner agent's current (total) reward, as well as the inner system's state (?).    
"""
class InnerSystem():
    def __init__(self, innerEnv, agent = None):
        
        self.innerEnv = innerEnv() # Inner environment.
        
        self.epsilon = 0.1 # Probability of selecting a random action.
        self.gamma = 0.99 # Discount factor.
        self.alpha = 0.5 # Learning rate.
        self.episodeCount = 1000 # Number of episodes.
        #self.windowSize = 10 # Number of steps to use for smoothing the plots...
        self.innerSystemReward = 0
        self.innerSystemState = []
        
        if agent is None:
            # Create an empty dictionary for each action.    
            self.Q = defaultdict(lambda: np.zeros(self.innerEnv.action_space.n))
        else:
            self.Q = agent
        # The epsilon-greedy policy to follow.
        self.policy = epsilon_greedy_policy(self.Q, self.epsilon, self.innerEnv.action_space.n)
    
        # Keep track of rewards and steps in each episode.
        self.episodeRewards = np.zeros(self.episodeCount)
        self.episodeSteps = np.zeros(self.episodeCount)
        
        
        # Might be moved to another class
        #self.low_state = np.array([-np.inf])
        #self.high_state = np.array([np.inf])
        # Instead of normalizing
        self.low_state = np.array([0])
        self.high_state = np.array([1])
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        
        # This is not general! Has to be generalized. Shall (hopefully) work for this experiment though.
        totalSpace = len(self.Q) * len(self.Q[0])
        spaceTuples_low = []
        spaceTuples_high = []
        #spaceTuples.append(spaces.Discrete(totalSpace))
        for entry in self.Q:
            for entryLength in range(0, self.Q[entry]):
                #spaceTuples_low.append(-np.inf)
                #spaceTuples_high.append(np.inf)
                # Instead of normalizing
                spaceTuples_low.append(0)
                spaceTuples_high.append(1)
        
        self.action_space = spaces.Box(low=spaceTuples_low, high=spaceTuples_high, shape=1)
        
        self.seed = self.seed(0)
    
    """
    For deterministic environment dynamics.
    """
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
    Reset the whole inner system. This includes resetting the inner environment,
    as well as the inner agent and the parameters defined for this system.
    """
    def reset(self):
        
        # Reset inner environment.
        self.innerEnv.reset()
        self.innerEnv.seed(self.seed)
        
        # Reset inner system parameters.
        self.epsilon = 0.1 # Probability of selecting a random action.
        self.gamma = 0.99 # Discount factor.
        self.alpha = 0.5 # Learning rate.
        self.episodeCount = 1000 # Number of episodes.
        self.windowSize = 10 # Number of steps to use for smoothing the plots...
        
        self.innerSystemReward = 0
        self.innerSystemState = []
        
        # Reset agent (for now, it's simply a Q-table).    
        self.Q = defaultdict(lambda: np.zeros(self.innerEnv.action_space.n))
        
        # Reset policy.
        self.policy = epsilon_greedy_policy(self.Q, self.epsilon, self.innerEnv.action_space.n)
    
        # Keep track of rewards and steps in each episode.
        self.episodeRewards = np.zeros(self.episodeCount)
        self.episodeSteps = np.zeros(self.episodeCount)
        
    
        # Reset the rest of the inner system.
        #self.lastaction = None
        self.innerSystemReward = 0        
        self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward)
    
    """
    This function is called every time the meta-learner performs an action in his own, outer system.
    At this point, this method shall make a whole round of interaction between the inner agent agent 
    and the inner environment. In other words, one step of the outer agent means several steps for the
    inner agent.
    Currently, it uses uses the Q-learning algorithm to returns the optimal action-value function Q, 
    a dictionary mapping state -> action values, following an epsilon-freedy policy.
    """
    def step(self, action, episodeCount, epsilon, gamma, alpha):
        self.episodeCount = episodeCount
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        
        for i in range(self.episodeCount):        
            # Reset environment
            innerState = self.innerEnv.reset()
            t = 0 # Number of steps in the current episode.
            
            while(True):
            
                # Move one step ahead.
                action_probs = self.policy(innerState)
                #print("Policy probabilities: ", action_probs)
                
                # Make a 'random.choice' from the available actions, taking into account 
                #the probabilities of each action that occur from the ε-greedy policy.
                action = np.random.choice(np.arange(len(action_probs)), p = action_probs)
                #print("Action: ", action)
                
                # Get next state, reward, done, and probability (= 1.0)
                # after performing the decided action.
                nextInnerState, innerReward, done, _ = self.innerEnv.step(action)
                
                # Update rewards and step count.
                self.episodeRewards[i] += innerReward
                self.episodeSteps[i] = t
                
                # Store total rewards at this point so as to be ready to use it when asked by the outer
                # system.
                self.innerSystemReward += innerReward
                
                # Q-learning rule.
                bestNextAction = np.argmax(self.Q[nextInnerState])
                self.Q[innerState][action] += self.alpha * (innerReward + self.gamma * self.Q[nextInnerState][bestNextAction] - self.Q[innerState][action])        
                
                if done:
                    #print("Inner episode rewards: {}, steps: {}".format(self.episodeRewards[i], self.episodeSteps[i]))
                    break
                
                t += 1
                innerState = nextInnerState
    
        
"""
Plot smoothed performance figures. These include episode rewards per step,
as well as number of steps per episodes.
"""
def PlotPerformanceFigures(windowSize, rewards, steps):

    print("Printing plots...")
    plt.figure(1)
    weights = np.repeat(1.0, windowSize) / windowSize
    episodeRewards_Smoothed = np.convolve(rewards, weights, 'valid')
    #episodeRewards_Smoothed = pd.Series(episodeRewards).rolling(windowSize, min_periods=windowSize).mean()
    plt.plot(episodeRewards_Smoothed, linewidth = 1.0)
    #plt.plot(episodeRewards, linewidth = 1.0)
    plt.figure(2)
    episodeSteps_Smoothed = np.convolve(steps, weights, 'valid')
    plt.plot(episodeSteps_Smoothed, linewidth = 1.0)
    #plt.plot(episodeSteps, linewidth = 1.0)
    
    #line.set_antialiased(False)
    plt.show()
    # Close the env and write monitor result info to disk
    #env.close()
    
"""  
Close inner environment (for practical reasons).
"""   
def CloseInnerEnvironment(self):
    self.innerEnv.close()
    
