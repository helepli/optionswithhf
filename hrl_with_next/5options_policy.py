import numpy as np
import optionInfos



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



