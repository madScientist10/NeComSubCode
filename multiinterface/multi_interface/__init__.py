from gym.envs.registration import register

register(
    id='multiinterface-v0',
    entry_point='multi_interface.envs:MultiInnerInterface',
)