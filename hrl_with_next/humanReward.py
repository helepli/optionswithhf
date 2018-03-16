import random
from optionInfos import *
# human reward functions

PUNISHMENT = -5.
PROBA = 1
INTERVENTIONS = 0

def humanreward(option, state, timestep, env):  # feedback of PUNISHMENT on wrong options, 0 otherwise, based on state
    global PROBA
    global INTERVENTIONS
    
    #or timestep % 20 == 0

    if option != -1 and random.random() < PROBA and timestep == 1 : # sort of feedback on the options. Only one feedback at the very beginning of the option.
        
        #PROBA *= 0.9997 # annealing
        INTERVENTIONS += 1
        print('Interventions so far', INTERVENTIONS, PROBA)
        
        # REAL HUMAN rewards
        #env.displayPosition() # display the agent's position in the grid to the human teacher
        #print("Option chosen: ", option)
        #punishment = input("Is the agent doing ok, yes(y) or no(n)?: ")
        #if punishment == "n": 
            #return (PUNISHMENT, None)
        #else: 
            #return (0, None) 

        
        
        #FAKE HUMAN rewards based on the agent's position
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
    
