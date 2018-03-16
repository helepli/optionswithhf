from gym.envs.registration import registry, register, make, spec



register(
    id='myGrid-v0',
    entry_point='gym_envs.myGrid:myGrid',
    kwargs={'y': 29, 'x': 27}
)

register(
    id='PuzzleRooms-v0',
    entry_point='gym_envs.6pieces:PuzzleRooms',
    kwargs={}
)
