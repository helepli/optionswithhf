import numpy as np
# common infos and functions to 5options_policy, 5options_shaping, humanProbas, humanReward


# the following three variables are taken by options.py:
nexts = {}
subs = {-1: [4, 5, 6, 7, 8], 0: range(4), 1: range(4), 2: range(4), 3: range(4), 4: range(4)}
num_options = 5
# 4 door-options (0, 1, 2, 3) : one for each door, to be taken in the direction of the goal. The 5th one is to go directly to the goal

height = 29
width = 27

OPTION_GOALS = [
    (width//3, (width-1)//2), # first door, top center --> option 0
    ((height-1)//2, (width-1)//3), # middle left --> opt 1
    ((height-1)//2, (2*width)//3), # middle right --> opt 2
    (((2*width)//3)+1, (width-1)//2), # last door, down center --> opt 3
    (height-1, width-1), # goal --> opt 4
]
# coordinates (y, x) of the 4 doors, and the goal ((-1, -1))
# The goal of the first option (n°0) is to get closer and closer to the first door, etc.


def distance_from_goal_xy(x, y, option):
    goal_y, goal_x = OPTION_GOALS[option]

    return np.sqrt((y - goal_y)**2 + (x - goal_x)**2)

def distance_from_goal(state, option): 
    y = state // width
    x = state % width

    return distance_from_goal_xy(x, y, option)
