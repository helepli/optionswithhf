

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
   
    return probas

def humanfeedback1(option, state, timestep):    
    if option != -1 and timestep == 1: # sort of feedback on the options. Only one feedback at the very beginning of the option.
        if option == 1 or option == 2:
            return (-0.5, None) # punishment for chosing options I don't like
        else:
            return (0, None) # good options, no feedback
    elif (timestep % 10) == 0:
    #if (timestep % 10) == 0:
        state_y = state // width
        state_x = state % width
        
        if state_y < width//3: 
            return (-0.1, None) # the agent is in the top room, we want it to go down  --> punishment
        
        elif state_y < ((2*width)//3)+1 and (state_x < (width-1)//3 or state_x > (2*width)//3): 
            return (-0.1, None) # the agent is in one of the middle rooms, left or right room, but not in the corridor leading 
        # to the goal room  --> punishment
        
        else:
            return (0, None) # the agent is in the corridor or in the "goal room" --> no feedback as positive feedback

    else:
        return (0, None)
    
    
def humanfeedback(option, state, timestep):  # feedback of -50 on wrong options, 0 otherwise, based on state
    if option != -1 and timestep == 1: # sort of feedback on the options. Only one feedback at the very beginning of the option.
        state_y = state // width
        state_x = state % width
        
        if state_y < width//3 and option != 0: # top room, we want it to choose option 0 (going to the first door) only
            return (-50, None)
        elif state_y < ((2*width)//3)+1:
            if (state_x > (width-1)//3 and state_x < (2*width)//3) and option != 3: # middle room, must take option taking the second door, going down
                return (-50, None)
            elif (state_x < (width-1)//3) and option != 1: # middle left room
                return (-50, None)
            elif (state_x > (2*width)//3) and option!= 2: # middle right room
                return (-50, None)
            else:
                return (0, None)
        else:
            return (0, None)
    else:
        return (0, None) # good options, no feedback
    
    
    
    
