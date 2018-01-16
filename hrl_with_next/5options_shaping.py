import numpy as np
import optionInfos


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
