from gym.envs.registration import register

register(
    id='innersys-mount-v0',
    entry_point='inner_sys_mount.envs:InnerSystemMount',
)