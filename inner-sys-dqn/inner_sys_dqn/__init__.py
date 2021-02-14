from gym.envs.registration import register

register(
    id='innersys-dqn-v0',
    entry_point='inner_sys_dqn.envs:InnerSystemDQN',
)