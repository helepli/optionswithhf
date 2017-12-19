from gym.envs.registration import registry, register, make, spec



register(
    id='myGrid-v0',
    entry_point='gym_envs.myGrid:myGrid',
    kwargs={'y': 29, 'x': 27}
)
