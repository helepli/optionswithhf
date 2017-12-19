import numpy as np

nexts = {}
subs = {-1: [4, 5, 6, 7, 8], 0: range(4), 1: range(4), 2: range(4), 3: range(4), 4: range(4)}
num_options = 5
# 4 door-options (0, 1, 2, 3) : one for each door, to be taken in the direction of the goal. The 5th one is to go directly to the goal

height = 29
width = 27
# this must be hardcoded...?

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



def policy(state, option):
    probas = [0.05] * 4 + [0.0] * 5 + [0.00] * 9    # 4 actions, 5 options, with or without termination 
    
    y = state // width
    x = state % width

    if option == -1:
        return None
    
        # Top-level policy, learned
        if y < OPTION_GOALS[0][0]:
            # High in the room, go to the first door
            probas[4 + 0] = 1.0 
        elif y < OPTION_GOALS[3][0]:
            # Middle room, go to the bottom center door
            probas[4 + 3] = 1.0
        else:
            # Bottom, go to the goal
            probas[4 + 4] = 1.0
    else:
        # hardcoded option policies
        # Return the action that goes the closest to the goal
        distances = [
            distance_from_goal_xy(x, y - 1, option) + 0.1,
            distance_from_goal_xy(x, y + 1, option) + 0.1,
            distance_from_goal_xy(x - 1, y, option),
            distance_from_goal_xy(x + 1, y, option),
        ]
        
        # Take the action that brings us the closest to the option goal
        end_proba = float(args.extra_args) if args.extra_args else 0.01 # fixed at best value
        probas[np.argmin(distances)] = 1.0 - end_proba
        probas[np.argmin(distances) + 9] = end_proba
        
        
    return probas

def shaping(option, old_state, new_state):
    if option == -1:
        # No shaping for the top-level policy (hint: insert human feedback here)
        return (0.0, None)
    elif option != 4:
        new_distance = distance_from_goal(new_state, option)
        return (0.1 * (distance_from_goal(old_state, option) - new_distance), new_distance == 0)    # Terminate the current option if the option goal has been reached
    else:
        # Option 4 has no shaping
        return (0.0, None)

