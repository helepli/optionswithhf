import random
from optionInfos import *

GIVING = 0.005 # proba og giving any advice to the agent

RIGHT = 0.6 # proba right option adviced by human
RANDOM = 0.0 # proba random option adviced by human
WRONG = 0.4 # proba wrong option adviced
INTERVENTIONS = 0

toplevel_policy = [
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '000000000000000000000000000',
    '-------------3-------------',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '111111113333333333322222222',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '11111111|333333333|22222222',
    '-------------4-------------',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
    '444444444444444444444444444',
]






def humanprobas(state, option): # policy shaping on options only, gives proba=1 to correct option given the situation  
    probas = [0.0] * 18      # 4 actions, 5 options, with or without termination 
    
    global GIVING
    global RIGHT
    global RANDOM
    global WRONG
    global INTERVENTIONS

    y = state // width
    x = state % width

    if option == -1 :
        
        coin = random.random()
        target = int(toplevel_policy[y][x])
        
        if coin < GIVING:
            
            advice = np.random.choice( ['right', 'random', 'wrong'], p=[RIGHT, RANDOM, WRONG])
            
            #RIGHT = RIGHT * 0.995 # annealing
        
            if advice == 'right':
                probas[4 + target] = 1.0
            
            elif advice == 'random': # put proba 1 to random option
                probas[random.randint(len(probas)-14, len(probas)-10)] = 1.0
            
            elif advice == 'wrong': # put proba 1 to anything but the right option
                probas[5] = 1.0 # always the same option, the human is lazy
                
            INTERVENTIONS += 1
            print('Interventions so far', INTERVENTIONS, GIVING)
            return probas
        
        else:
            return None

    return None


    
    
    
