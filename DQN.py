import gym
from gym.utils import seeding
from gym import spaces
import numpy as np


from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

env = gym.make('Acrobot-v1')


del model # remove to demonstrate saving and loading

model = DQN.load("deepq_Acrobot")

obs = env.reset()

    
"""
Compute the inner system's current state, according to the inner environment's total rewards at this point.
"""
def GetCurrentInnerSystemState(reward, lastAgentVersion):
    return [reward, lastAgentVersion]


"""
Inner system within the meta-learning process, implemented with a Q-Learning algorithm.
The inner system consists of the inner (input) environment, as well as the agent,
and the rest of parameters that defined the interaction between the two.
These include the outer reward that this system shall send to the outer system (meta-learner)
as well as the current state communicated to the meta-learner. This state is a vector
composed by the inner agent's current (total) reward, as well as the inner system's state (?).    
"""
class InnerSystemDQN(gym.Env):
    def __init__(self, innerEnv, agent = None):
        
        self.innerEnv = innerEnv() # Inner environment.
        
        self.epsilon = 0.1 # Probability of selecting a random action.
        self.gamma = 0.99 # Discount factor.
        self.alpha = 0.5 # Learning rate.
        self.episodeCount = 10 # Number of episodes.

        self.innerSystemReward = 0
        self.innerSystemState = []
        
        directory = "DQN"
        self.logDir = "./ppo2_innersys/" + directory
        
        self.model = DQN(MlpPolicy, self.innerEnv, verbose=1, tensorboard_log=self.logDir)
        # Keep track of rewards and steps in each episode.
        #self.episodeRewards = np.zeros(self.episodeCount)
        #self.episodeSteps = np.zeros(self.episodeCount)
        
# =============================================================================
# ----------        OLD         -------------------
#         Might be moved to another class
#         # Currently, the state vector is actually the reward.   
#         self.low_state = np.array([-np.inf])
#         self.high_state = np.array([np.inf])
# =============================================================================
        
        
        
        # Crease observation space (which has the same form as the weight vector of the trained DQN agent).
        # Check for more details in action space creation below (is same as observation space creation).
        for weight in self.model.get_weights():
            self.low_state = np.append(self.low_state, -np.inf)
            self.high_state = np.append(self.high_state, np.inf)       
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        
        
        # This is not general! Has to be generalized. Shall (hopefully) work for this experiment though.
        # An action in this environment means to generate a new DQN agent. Therefore, the
        # action space has the same form of the weight vector of a DQN agent
        # (i.e. a -np.inf (low) and an np.inf (high))
        self.low_action = []
        self.high_action = np.array([])
        #spaceTuples.append(spaces.Discrete(totalSpace))
        for weight in self.model.get_weights():
            self.low_action = np.append(self.low_action, -np.inf)
            self.high_action = np.append(self.high_action, np.inf)
        
        self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
        
        self.seed = self.seed(0)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    """
    Reset the whole inner system. This includes resetting the inner environment,
    as well as the inner agent and the parameters defined for this system.
    """
    def reset(self, episodeCount = 10, epsilon = 0.1, gamma = 0.99, alpha = 0.5):
        
        # Reset inner environment.
        self.innerEnv.reset()
        
        # Reset inner system parameters.
        self.epsilon = epsilon # Probability of selecting a random action.
        self.gamma = gamma # Discount factor.
        self.alpha = alpha # Learning rate.
        self.episodeCount = episodeCount # Number of episodes.
        #self.windowSize = 10 # Number of steps to use for smoothing the plots...
        
        self.innerSystemReward = 0
        self.innerSystemState = []

        
        # Reset agent (for now, it's simply a Q-table).    
        self.model = DQN(MlpPolicy, self.innerEnv, verbose=1, tensorboard_log=logDir)
        #for state in range(0,self.innerEnv.observation_space.n):
            #self.Q[state] = np.zeros(self.innerEnv.action_space.n)

    
        # Keep track of rewards and steps in each episode.
        #self.episodeRewards = np.zeros(self.episodeCount)
        #self.episodeSteps = np.zeros(self.episodeCount)
        
    
        # Reset the rest of the inner system.
        self.innerSystemReward = 0        
        self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.model.get_weights())
        
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
        self.innerEnv.reset()
        
        self.model.set_weights(action)
        self.model.learn(total_timesteps=25000)
        self.model.save("deepq_Acrobot")        
        
        while done == False:
            action, _states = model.predict(obs)
            innerState, self.innerSystemReward, done, info = self.innerEnv.step(action)
            self.innerSystemState = GetCurrentInnerSystemState(self.innerSystemReward, self.model.get_weights())
        #PlotPerformanceFigures(1, self.episodeRewards, self.episodeSteps)
        return (self.innerSystemState, self.innerSystemReward, done, info)
                

# =============================================================================
#     def GetCurrentAgent(self):
#         return self.Q
#     
#     def SetCurrentAgent(self, agent):
#         self.Q = agent
# =============================================================================
        
        
"""  
Close inner environment (for practical reasons).
"""   
def CloseInnerEnvironment(self):
    self.innerEnv.close()