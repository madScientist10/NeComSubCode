import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('FATAL')
tf.autograph.set_verbosity(tf.logging.FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)  

import gym
import numpy as np
import pandas as pd
from EpsilonGreedyPolicy import epsilon_greedy_policy
from collections import defaultdict
import matplotlib.pyplot as plt
#from InnerSystem import InnerSystem

from InnerEnv import CliffWalkingEnv
from InnerEnvMount import MountainCarEnv
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym.envs.toy_text.taxi import TaxiEnv
from gym.envs.classic_control import AcrobotEnv
#from gym.envs.box2d import BipedalWalkerEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, ACKTR, A2C, TD3, DQN
from MultiInnerInterface import MultiInnerInterface

from stable_baselines.common.callbacks import BaseCallback

#env = GridworldEnv()
# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
outdir = '/tmp/random-agent-results'

#env = wrappers.Monitor(env, directory=outdir, force=True)
#env.seed(0)


#innerEnv1 = CliffWalkingEnv


#innerEnv = CliffWalkingEnv

"""
De-register environment from previous run (so as not to restart). Quick fix.
Might be moved elsewhere (e.g. as innerSystem method)
"""
# =============================================================================
# # =============================================================================
# env_dict = gym.envs.registration.registry.env_specs.copy()
# for env in env_dict:
#      if 'multiinterface-v0' in env:
#           print('Remove {} from registry'.format(env))
#           del gym.envs.registration.registry.env_specs[env]
# # =============================================================================
# =============================================================================
#import gym_sokoban
# =============================================================================
# innersysmount_name = 'inner_sys_mount'
# innersysmount_id = 'innersys-mount-v0'
# #innersysmount_string =  innersysmount_name + ':' + innersysmount_id
# innersysmount_env = MountainCarEnv
# =============================================================================

#innerEnvVecName = ['inner_sys', 'inner_sys_mount']
#innerEnvVecName = ['inner_sys', 'inner_sys']
innerEnvVecName = ['inner_sys_dqn', 'inner_sys_dqn']
#innerEnvVecId = ['innersys-v0', 'innersys-mount-v0']
#innerEnvVecId = ['innersys-v0', 'innersys-v0']
innerEnvVecId = ['innersys-dqn-v0', 'innersys-dqn-v0']
#innerEnvVec = [CliffWalkingEnv, MountainCarEnv]
innerEnvVec = ['LunarLander-v2', 'MountainCar-v0']
#innerEnvVec = ['Acrobot-v1', 'CartPole-v1']
#innerEnvVec = ['CliffWalking-v0', 'CartPole-v1']
#innerEnvVec = ['CliffWalking-v0', 'PongNoFrameskip-v4']
#innerEnvVec = ['PongNoFrameskip-v4', 'Asteroids-v0']
arg = [[innerEnvVecName[1]], [innerEnvVecId[1]], [innerEnvVec[1]]]
#arg = [innerEnvVecName, innerEnvVecId, innerEnvVec]
# =============================================================================
# innerEnvVecName = [innersysmount_name]
# innerEnvVecId = [innersysmount_id]
# innerEnvVec = [MountainCarEnv]
# =============================================================================

#env = MultiInnerInterface(innerEnvVecName, innerEnvVecId, innerEnvVec)

    #env = gym.make('multi_interface:multiinterface-v0', innerVec = arg, differentAlgo = False)
    #env = DummyVecEnv([lambda: env])
    #env = VecCheckNan(env, raise_exception=True)
    #env = gym.make('inner_sys_mount:innersys-mount-v0', innerEnv = MountainCarEnv) # Environment.
    #env = gym.make('inner_sys:innersys-v0', innerEnv = CliffWalkingEnv) # Environment.


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, env, verbose=1):
        self.is_tb_set = False
        self.env = env
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
# =============================================================================
#         # Log additional tensor
#         if not self.is_tb_set:
#             with self.model.graph.as_default():
#                 tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
#                 self.model.summary = tf.summary.merge_all()
#             self.is_tb_set = True
# =============================================================================
        # Log scalar value (here a random variable)
        value = self.env.readyEnvs[0].oldReward
        summary = tf.Summary(value=[tf.Summary.Value(tag='inner_reward', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True




OuterTotalTimesteps = 10000
InnerTotalTimesteps = 1000000
NumberOfBatches = 100
Seed = 40

outerAlgos = ['A2C', 'PPO']
outerAlgo = outerAlgos[1]

if outerAlgo == 'PPO':
    #PPO
    lr = 0.0022
    nSteps = 32
    entCoef = 0.01
    nMinibatches = 4
    noptEpochs = 4
    directorySubstring = "-lr_" + str(lr) + "-nSteps_" + str(nSteps) + "-entCoef_" + str(entCoef) + \
        "-nMiniBatches_" + str(nMinibatches) + "-noptepochs_" + str(noptEpochs)
else:
    #A2C
    lr = 0.95 #0.0006
    #ent_coef = 0.01
    directorySubstring = "-lr_" + str(lr)

test = True
qlearning = False
if test == True:
    print('Testing phase in progress. Hold on to your seats...')
    
    directory = "GraphAcro_Test-OuterAlgo_" + outerAlgo + "-Batches_" + str(NumberOfBatches) + \
              directorySubstring + "-Seed_" + str(Seed) + "-OuterSteps_" + str(OuterTotalTimesteps) + str(arg[2]) + "/"
    logDir = "./skatopaper/" + directory
    #dummyVecEnv = DummyVecEnv(env)
    #VecCheckNan(dummyVecEnv, raise_exception=True, warn_once=True, check_inf=True)
    #model = PPO2(MlpPolicy, env, verbose=1, n_steps=16, ent_coef=0, nminibatches=4, noptepochs=4, tensorboard_log=logDir)
    
    ### DONT - IT'S TOO MEMORY HUNGRY FOR MY CUSTOM ENVS ###
    #model = ACKTR(MlpPolicy, env, verbose=1, n_steps=20, ent_coef=0,  tensorboard_log=logDir)
    ###
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if 'multiinterface-v0' in env:
            print('Removing {} from registry'.format(env))
            del gym.envs.registration.registry.env_specs[env]
    from MultiInnerInterface import MultiInnerInterface
    out_batch = 0
    env_args ={ 'innerVec': arg, 'outBatch': out_batch, 'noBatches': False, 'differentAlgo' : False}
    #env = gym.make('multi_interface:multiinterface-v0', innerVec = arg, outBatch = out_batch, noBatches= True, differentAlgo = False)
    env = gym.make('multi_interface:multiinterface-v0', env_args = env_args)

    #env = make_vec_env(gym.make(), env_kwargs = env_args, n_envs=4)
    
    if qlearning == True:
        env.reset()
        env.seed(Seed)
        
        
        model = A2C(MlpPolicy, env, verbose=1, tensorboard_log=logDir, seed = Seed)
            #model = A2C(MlpPolicy, env, learning_rate = 0.0006, ent_coef=0.01, verbose=1, tensorboard_log=logDir, seed = Seed)
        model.learn(total_timesteps=OuterTotalTimesteps)
    
    else:
        
        env.reset()
        env.seed(Seed)
        #model = PPO2(MlpPolicy, env, verbose=1, learning_rate=lr, n_steps=nSteps, ent_coef=entCoef, nminibatches=nMinibatches, noptepochs=noptEpochs, tensorboard_log=logDir)
        model = PPO2(MlpPolicy, env, n_steps = 32, nminibatches = 1, lam = 0.8, 
                 gamma = 0.98, noptepochs = 20, ent_coef = 0.0, learning_rate = 0.001, cliprange=0.2,
                 tensorboard_log=logDir, seed = Seed)
        #model = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.0022, n_steps=32, ent_coef=0.01, nminibatches=4, noptepochs=4, tensorboard_log=logDir)
        #model = A2C(MlpPolicy, env, learning_rate = lr, verbose=1, tensorboard_log=logDir, seed = Seed)
        #model = A2C(MlpPolicy, env, learning_rate = 0.95, verbose=1, tensorboard_log=logDir, seed = Seed)
        #model = A2C(MlpPolicy, env, learning_rate = 0.0006, ent_coef=0.01, verbose=1, tensorboard_log=logDir, seed = Seed)
            
    
        for out_batch in range(1, 2):
            
            #env.reset()
            #env.seed(Seed)
            if NumberOfBatches == 1:
                startBatchVector = 0
                endBatchVector = len(env.readyEnvs[0].modelWeightVector)
            else:
                outBatch = out_batch * int((len(env.readyEnvs[0].modelWeightVector) - 1) / NumberOfBatches)
                startBatchVector = outBatch + 1
                endBatchVector = outBatch + int((len(env.readyEnvs[0].modelWeightVector) - 1) / NumberOfBatches)
            
            env.readyEnvs[0].startBatchVector = startBatchVector
            env.readyEnvs[0].endBatchVector = endBatchVector
            #model = PPO2(MlpPolicy, env, verbose=1, learning_rate=0.22, n_steps=128, ent_coef=0, nminibatches=4, noptepochs=4, tensorboard_log=logDir)
            #model = A2C(MlpPolicy, env, learning_rate = 0.15, verbose=1, tensorboard_log=logDir, seed = Seed)
            #model = A2C(MlpPolicy, env, learning_rate = 0.0006, ent_coef=0.01, verbose=1, tensorboard_log=logDir, seed = Seed)
            model.learn(total_timesteps=OuterTotalTimesteps, callback=TensorboardCallback(env))
            model.learn(total_timesteps=OuterTotalTimesteps)
            
            
            # Enjoy trained agent
            predictThen = False
            if predictThen == True:
                from stable_baselines.deepq.policies import MlpPolicy
                env = gym.make('CartPole-v1')
                ass = DQN.load("deepq", env =env, tensorboard_log=logDir)
                
                #ass.learn(total_timesteps=InnerTotalTimesteps)
                  
                eps = 100
                temp_reward = np.zeros(eps)
                for ep in range(eps):
                    done = False
                    innerState = env.reset()
                    count = 0
                    while done == False:
        # =============================================================================
        #                 if count == 198:
        #                     print("ti ginetai gamw ti mana tou")
        # =============================================================================
                        innerAction, _states = ass.predict(innerState)
                        innerState, innerReward, done, info = env.step(innerAction)
                        # Below is the sum of rewards that keep coming _done_ flag is raised.
                        # This is due to the nature of the environment.
                        temp_reward[ep] = temp_reward[ep] + innerReward
                        count = count + 1
                print("Ep. {0} Average score: {1}".format(ep, sum(temp_reward) / (eps)))
                
            
        
else:
    OuterTotalTimesteps = 300000
    directory = "A2CMountaEval-OuterAlgo_" + outerAlgo + directorySubstring + "-Seed_" + str(Seed) + \
              "-OuterSteps_" + str(OuterTotalTimesteps) + "_CartPole-v1/"
    logDir = "./skatopaperMount/" + directory

    from stable_baselines.common.atari_wrappers import make_atari
    #from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
    from stable_baselines.common import make_vec_env
    #env = make_vec_env('CartPole-v1', n_envs=1)
    #env = make_vec_env('CartPole-v1', n_envs=4)
    env = gym.make('MountainCar-v0')
    #env = gym.make('Acrobot-v1')
    #env = gym.make('PongNoFrameskip-v4')
    #env = make_atari('BreakoutNoFrameskip-v4')
    
    #model = ACKTR(MlpPolicy, env, verbose=1, tensorboard_log=logDir, n_steps=20, seed = 0, learning_rate=0.25, vf_coef=0.25)
    #from stable_baselines.deepq.policies import MlpPolicy
    #model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=logDir, seed = Seed)
    #model = PPO2(MlpPolicy, env, n_steps = 16, nminibatches = 8, lam = 0.98, gamma = 0.99, noptepochs = 4, ent_coef = 0.0, tensorboard_log=logDir, seed = Seed)

    #from stable_baselines.td3.policies import MlpPolicy
    #from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    #n_actions = env.action_space.shape[-1]
    #action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    #model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1, tensorboard_log=logDir, seed=Seed)
    
    model = A2C(MlpPolicy, env, ent_coef=0, verbose=1, tensorboard_log=logDir, seed = Seed)
    #model = DQN(MlpPolicy, env, verbose=1, seed = Seed, tensorboard_log=logDir)
    #model = PPO2(MlpPolicy, env, verbose=1,learning_rate=0.00015, n_steps=16, ent_coef=0, nminibatches=4, noptepochs=4, tensorboard_log=logDir)

    model.learn(total_timesteps=OuterTotalTimesteps)
    #model.save("deepq_cartpole")
    
    #del model # remove to demonstrate saving and loading
    
    #model = DQN.load("deepq_cartpole")
    
    obs = env.reset()
    
# =============================================================================
# =============================================================================
#     count = 0
    eps = 1000
    reward = np.zeros(eps)
    for ep in range(eps):
        reward[ep] = 0
        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            reward[ep] = rewards + reward[ep]
            if dones == True:
                print(reward[ep])
                #print(ep)
                break;
            #env.render()
#         count = count + 1
#     print(count)
# =============================================================================
# =============================================================================
    



    
    
# =============================================================================
#     obs = env.reset(episodeCount = 20)
#     currentAgent = env.GetCurrentAgent()
#     for i in range(5):
#         env.SetCurrentAgent(currentAgent)
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = env.step(action)
#         print(rewards)
#         currentAgent = env.GetCurrentAgent()
#         #env.render()
# =============================================================================