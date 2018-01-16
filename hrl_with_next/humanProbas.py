import random
import optionInfos


def humanProbas(state, option): # for policy shaping with simulated human feedback
    probas = [0.05] * 4 + [0.0] * 5 + [0.00] * 9    # 4 actions, 5 options, with or without termination 
    
    y = state // width
    x = state % width

    if option == -1:
        # Top-level policy, advice on options
        if y < OPTION_GOALS[0][0]:
            # High in the room, go to the first door
            probas[4 + 0] = 1.0 
        elif y < OPTION_GOALS[3][0]:
            # Middle room, go to the bottom center door
            probas[4 + 3] = 1.0
        else:
            # Bottom, go to the goal
            probas[4 + 4] = 1.0
    
    if random.random() < 0.1:
        return probas
    else:
        return None


    
    
    
