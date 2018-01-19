from optionInfos import *

option_policies = [
    [
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRBLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRbLLLLLLLLLLLLL',
        '-------------B-------------',
        '        |RRRRtLLLL|        ',
        '        |RRRRTLLLL|        ',
        '        |RRRRTLLLL|        ',
        '        |RRRRTLLLL|        ',
        '        RRRRRTLLLLL        ',
        '        |RRRRTLLLL|        ',
        '        |RRRRTLLLL|        ',
        '        |RRRRTLLLL|        ',
        '        |RRRRTLLLL|        ',
        '-------------T-------------',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
    ],
    [
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '-------------B-------------',
        'BBBBBBBB|BBBBBBBBB|        ',
        'BBBBBBBB|BBBBBBBBB|        ',
        'BBBBBBBB|BBBBBBBBB|        ',
        'BBBBBBBB|BBBBBBBBB|        ',
        'RRRRRRRrRlLLLLLLLLL        ',
        'TTTTTTTT|TTTTTTTTT|        ',
        'TTTTTTTT|TTTTTTTTT|        ',
        'TTTTTTTT|TTTTTTTTT|        ',
        'TTTTTTTT|TTTTTTTTT|        ',
        '-------------T-------------',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
    ],
    [
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '-------------B-------------',
        '        |BBBBBBBBB|BBBBBBBB',
        '        |BBBBBBBBB|BBBBBBBB',
        '        |BBBBBBBBB|BBBBBBBB',
        '        |BBBBBBBBB|BBBBBBBB',
        '        RRRRRRRRRrLlLLLLLLL',
        '        |TTTTTTTTT|TTTTTTTT',
        '        |TTTTTTTTT|TTTTTTTT',
        '        |TTTTTTTTT|TTTTTTTT',
        '        |TTTTTTTTT|TTTTTTTT',
        '-------------T-------------',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
    ],
    [
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '-------------B-------------',
        '        |RRRRBLLLL|        ',
        '        |RRRRBLLLL|        ',
        '        |RRRRBLLLL|        ',
        '        |RRRRBLLLL|        ',
        '        RRRRRBLLLLL        ',
        '        |RRRRBLLLL|        ',
        '        |RRRRBLLLL|        ',
        '        |RRRRBLLLL|        ',
        '        |RRRRbLLLL|        ',
        '-------------B-------------',
        'RRRRRRRRRRRRRtLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
        'RRRRRRRRRRRRRTLLLLLLLLLLLLL',
    ],
    [
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '                           ',
        '------------- -------------',
        '        |         |        ',
        '        |         |        ',
        '        |         |        ',
        '        |         |        ',
        '                           ',
        '        |         |        ',
        '        |         |        ',
        '        |         |        ',
        '        |         |        ',
        '-------------B-------------',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBB',
        'BBBBBBBBBBBBBBBBBBBBBBBBBBb',
        'RRRRRRRRRRRRRRRRRRRRRRRRRrR',
    ],
]


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
        command = option_policies[option][y][x]

        if command == ' ':
            # No policy defined here, act randomly and terminate
            probas[9:13] = [1.0] * 4
        else:
            terminate = (command in 'tblr')
            action_index = {
                'T': 0,
                'B': 1,
                'L': 2,
                'R': 3
            }[command.upper()]

            if terminate:
                probas[action_index + 9] = 1.0  # Execute the correct action and terminate right after the goal
            else:
                probas[action_index] = 1.0

    return probas



