import optionInfos
# human reward functions


def humanreward1(option, state, timestep):    
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
    
    
def humanreward(option, state, timestep):  # feedback of -50 on wrong options, 0 otherwise, based on state
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
    
