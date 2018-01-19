from optionInfos import *
# human reward functions

PUNISHMENT = -5.


def humanreward(option, state, timestep):  # feedback of PUNISHMENT on wrong options, 0 otherwise, based on state
    if option != -1 and timestep == 1: # sort of feedback on the options. Only one feedback at the very beginning of the option.
        state_y = state // width
        state_x = state % width
        
        if state_y < width//3 and option != 0: # top room, we want it to choose option 0 (going to the first door) only
            return (PUNISHMENT, None)
        elif state_y < ((2*width)//3)+1:
            if (state_x > (width-1)//3 and state_x < (2*width)//3) and option != 3: # middle room, must take option taking the second door, going down
                return (PUNISHMENT, None)
            elif (state_x < (width-1)//3) and option != 1: # middle left room
                return (PUNISHMENT, None)
            elif (state_x > (2*width)//3) and option != 2: # middle right room
                return (PUNISHMENT, None)
            else:
                return (0, None)
        else:
            return (0, None)
    else:
        return (0, None) # good options, no feedback
    
