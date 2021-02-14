from gym.envs.registration import register

register(
    id='innersys-v0',
    entry_point='inner_sys.envs:InnerSystem',
)