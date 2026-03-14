# Need:
# - Open list (implemented as a priority queue)
# - Closed list (implemented as a set)
# - Parent dictionary (to reconstruct the path)
# - four move functions (up, down, left, right)
# - breadth first search function
# - 8 bit puzzle 

import numpy as np
import sys
open_list = []
closed_list =[]
final_path = []

node_index = 0

parents = {}   # child_tuple -> parent_tuple (or None)
state_to_idx = {}   # child_tuple -> parent node index 

#----------Define Function to Locate the Blank Space-----------------------   
def find_blank_space(state):
    """
    Assume that the state is given as a vector, where the entries correspond as follows:
    [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
    """
    state_mtx = state.reshape(3, 3).copy()
    # print(f"starting with \n {state_mtx}")
    for i in range(len(state_mtx[0])):
        for j in range(len(state_mtx)): 
            # print(state_mtx[i, j])
            if state_mtx[i, j] != 0:
                pass
            elif state_mtx[i, j] == 0:
                return i, j

#----------Define Function to Evaluate Possible Moves-----------------------   
def generate_possible_moves(state):
    """
    once the location of the blank space is located, this function will 
    return the valid possible moves 
    """
    valid = []
    i, j = find_blank_space(state)
    if i != 0:
        valid.append("up")
    if i != 2:
        valid.append("down")
    if j != 0:
        valid.append("left")
    if j != 2:
        valid.append("right")

    return valid, i, j
        
#----------Define Functions to Move Up, Down, Left and Right-----------------------
def move_up(state, i, j):
    state = state.reshape(3, 3).copy()
    i_init = i
    j_init = j
    if i != 0:
        state[i, j] = state[i_init-1, j_init]
        state[i-1, j] = 0
        return state
    else:
        print("Unable to Move Up")
        state = None
        
def move_down(state, i, j):
    state = state.reshape(3, 3).copy()
    i_init = i
    j_init = j
    if i != 2:
        state[i, j] = state[i_init+1, j_init]
        state[i+1, j] = 0
        return state
    else:
        print("Unable to Move Down")
        state = None

def move_left(state, i, j):
    state = state.reshape(3, 3).copy()
    i_init = i
    j_init = j
    if j != 0:
        state[i, j] = state[i_init, j_init-1]
        state[i, j-1] = 0
        return state
    else:
        print("Unable to Move Left")    
        state = None   
        return

def move_right(state, i, j):
    state = state.reshape(3, 3).copy()
    i_init = i
    j_init = j
    if j != 2:
        state[i, j] = state[i_init, j_init+1]
        state[i, j+1] = 0  
        return state
    else:
        print("Unable to Move Right")
        state = None
        return


#----------Define Function to Expand the Search Tree----------------------
def expand_tree(prev_state, goal_state, info):
    """
    Evaluate the valid moves for each of the states that were just evaluated against the goal state. 
    If any of the valid moves have not been seen before, add them to the open list and parent dictionary. 
    Once all of the valid moves have been evaluated, run the next round of comparisons against the goal state.
    """
    global node_index
    print(f" open states: {len(info)}")
    
    # iterate through the states that were just evaluated and the corresponding valid moves and blank space locations
    for (prev_state, prev_idx), info in zip(prev_state, info):

        # unpack the valid moves and the location of the blank space for the state that was just evaluated
        valid, (i, j) = info

        if "up" in valid:
            state = move_up(prev_state, i, j)

            child_t = tuple(state.reshape(-1))
            if child_t not in parents:
                parents[child_t] = (tuple(prev_state.reshape(-1)), node_index, prev_idx)
                state = (state.reshape(-1))
                open_list.append((state, node_index))
                node_index += 1

        if "down" in valid:
            state = move_down(prev_state, i, j)

            child_t = tuple(state.reshape(-1))
            if child_t not in parents:
                parents[child_t] = (tuple(prev_state.reshape(-1)), node_index, prev_idx)
                state = (state.reshape(-1))
                open_list.append((state, node_index))
                node_index += 1

        if "right" in valid:
            state = move_right(prev_state, i, j)

            child_t = tuple(state.reshape(-1))
            if child_t not in parents:
                parents[child_t] = (tuple(prev_state.reshape(-1)), node_index, prev_idx)
                state = (state.reshape(-1))
                open_list.append((state, node_index))
                node_index += 1

        if "left" in valid:
            state = move_left(prev_state, i, j)

            child_t = tuple(state.reshape(-1))
            if child_t not in parents:
                parents[child_t] = (tuple(prev_state.reshape(-1)), node_index, prev_idx)
                state = (state.reshape(-1))
                open_list.append((state, node_index))
                node_index += 1
                
    # move onto the next round of comparisons against the goal state
    compare_against_goal(goal_state)

#----------Define Function to Evaluate if the Goal Position has been Reached----------------------
def compare_against_goal(goal_state):
    """
    Go through the open list and see if any of them match the goal state. if not, move on to next set of possible moves
    # """
    global node_index

    # initialize objects to store future moves once comparison is complete
    info = []
    prev_state = []

    # go through all entries until open list is empty
    while open_list:
        # go through the open list one by one
        state = open_list.pop(0)
        key = tuple(state[0])

        # check to see if the goal is in the open list
        if np.array_equal((state[0]), (goal_state)):
            final_state = state[0]
            final_path.append(final_state)
            print("GOAL REACHED!")
            generate_path(final_state)
            sys.exit()

        # add the ones currently being evaluated to matrix that holds the next tree expansions
        if key not in closed_list:
            closed_list.append(key)

            # add the state that was just evaluated to the list of states that will be expanded in the next round
            prev_state.append(state)

            # now, generate the possible moves for the state that was just evaluated
            valid, i, j = generate_possible_moves(state[0])
            update = (valid, (i, j))

            # store the inforamtion for all of the states that wer evaluated in this round
            info.append(update)

    # run next round
    expand_tree(prev_state, goal_state, info)

#----------Define Function to Execute BFS----------------------
def breadth_first_search(start_state, goal_state):
    """
    Initialize the BFS by adding the start state to the open list and then 
    calling the function to compare against the goal state. The BFS will continue 
    until the goal is reached, at which point the path to the goal will be generated.
    """
    global node_index

    # initialize the parent dictionary with the start state (which has no parent)
    child_t = tuple(start_state.reshape(-1))
    parents[child_t] = (None, node_index, None)

    # add first state to the open list
    open_list.append((start_state, node_index))
    node_index += 1 

    # start loop
    compare_against_goal(goal_state)

# -------------once the goal is reached, re-construct the path to get there-----------------------
def generate_path(final_state):
    """
    Once the goal state is reached, re-create the file path
    using the Parents dictionary
    """
    print("Generating Path to Goal...")
    print(final_state)
    child = final_state

    order = [tuple(child)]

    # reconstruct the path by going through the parent dictionary until we reach the start state (which has no parent)
    while parents[tuple(child)][0] is not None:
        child = parents[tuple(child)][0] 
        order.append(child)

    # reverse the order
    order.reverse()

    # Transpose the output to match the style used in the slides
    for i in range(len(order)):
        order[i] = np.asarray(order[i]).reshape(3, 3)
        order[i] = np.transpose(order[i])

    generate_text_files(order)
    
#----------Define Function to generate the desired text files----------------------
def generate_text_files(order):
    filename_1 = "Nodes.txt"
    explored_states = parents.keys()
    # (par_t, child_idx, par_idx) = parents.values()
    try:
        with open(filename_1, 'w') as file:
            for row in explored_states:
                file.write(' '.join(map(str, row)) + '\n')

    except IOError as e:
        print(f"Error writing to file: {e}")
    finally:
        file.close()

    filename_2 = "NodesInfo.txt"    
    try:
        with open(filename_2, 'w') as file:
            file.write("Node Index | Parent Node Index | Node\n")
            file.write("--------------------------------------\n")

            # parents: child_t -> (par_t, child_idx, par_idx)
            rows = []
            for child_t, (par_t, child_idx, par_idx) in parents.items():
                file.write(f"{child_idx} \t {par_idx} \t" + ' '.join(map(str, child_t)) + '\n')
            
    except IOError as e:
        print(f"Error writing to file: {e}")
    finally:
        file.close()


    filename_3 = "nodePath.txt"
    for i in range(len(order)):
        order[i] = np.asarray(order[i]).reshape(-1)

    try:
        with open(filename_3, 'w') as file:
            # parents: child_t -> (par_t, child_idx, par_idx)
            for state in order:                      # order = path from start → goal
                flat = np.asarray(state).reshape(-1) # 1×9
                file.write(" ".join(map(str, flat)) + "\n")
            
    except IOError as e:
        print(f"Error writing to file: {e}")
    finally:
        file.close()

#----------Run BFS----------------------
if __name__ == "__main__":
    start_state = np.array([7, 5, 4, 0, 3, 2, 8, 1, 6])
    goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])

    breadth_first_search(start_state, goal_state)


#-----------------------------------------
# # REVISED:
# import numpy as np
# import sys
# from collections import deque

# open_list = deque()
# closed_set =set()
# final_path = []

# node_index = 0

# parents = {}   # child_tuple -> parent_tuple (or None)
# state_to_idx = {}   # child_tuple -> parent node index 

# #----------Define Function to Locate the Blank Space-----------------------   
# def find_blank_space(state):
#     """
#     Assume that the state is given as a vector, where the entries correspond as follows:
#     [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
#     """
#     state_mtx = state.reshape(3, 3).copy()
#     # print(f"starting with \n {state_mtx}")
#     for i in range(len(state_mtx[0])):
#         for j in range(len(state_mtx)): 
#             # print(state_mtx[i, j])
#             if state_mtx[i, j] != 0:
#                 pass
#             elif state_mtx[i, j] == 0:
#                 return i, j

# #----------Define Function to Evaluate Possible Moves-----------------------   
# def generate_possible_moves(state):
#     """
#     once the location of the blank space is located, this function will 
#     return the valid possible moves 
#     """
#     valid = []
#     i, j = find_blank_space(state)
#     if i != 0:
#         valid.append("up")
#     if i != 2:
#         valid.append("down")
#     if j != 0:
#         valid.append("left")
#     if j != 2:
#         valid.append("right")

#     return valid, i, j
        
# #----------Define Functions to Move Up, Down, Left and Right-----------------------
# def move_up(state, i, j):
#     state = np.asarray(state).reshape(3, 3).copy()
#     # print(f"STATE: {state}")

#     i_init = i
#     j_init = j
#     if i != 0:
#         state[i, j] = state[i_init-1, j_init]
#         state[i-1, j] = 0
#         return state
#     else:
#         print("Unable to Move Up")
#         state = None
        
# def move_down(state, i, j):
#     state = np.asarray(state).reshape(3, 3).copy()
#     i_init = i
#     j_init = j
#     if i != 2:
#         state[i, j] = state[i_init+1, j_init]
#         state[i+1, j] = 0
#         return state
#     else:
#         print("Unable to Move Down")
#         state = None

# def move_left(state, i, j):
#     state = np.asarray(state).reshape(3, 3).copy()
#     i_init = i
#     j_init = j
#     if j != 0:
#         state[i, j] = state[i_init, j_init-1]
#         state[i, j-1] = 0
#         return state
#     else:
#         print("Unable to Move Left")    
#         state = None   
#         return

# def move_right(state, i, j):
#     state = np.asarray(state).reshape(3, 3).copy()
#     i_init = i
#     j_init = j
#     if j != 2:
#         state[i, j] = state[i_init, j_init+1]
#         state[i, j+1] = 0  
#         return state
#     else:
#         print("Unable to Move Right")
#         state = None
#         return


# #----------Define Function to Evaluate if the Goal Position has been Reached----------------------
# def compare_against_goal(goal_state):
#     """
#     Go through the open list and see if any of them match the goal state. if not, move on to next set of possible moves
#     # """
#     global node_index

#     # go through all entries until open list is empty
#     while open_list:
#         # go through the open list one by one
#         state, curr_idx = open_list.popleft()
#         key = tuple(state)

#         # check to see if the goal is in the open list
#         if np.array_equal((state), (goal_state)):
#             final_state = state
#             final_path.append(final_state)
#             print("GOAL REACHED!")
#             generate_path(final_state)
#             sys.exit()

#         # add the ones currently being evaluated to matrix that holds the next tree expansions
#         if key  in closed_set:
#             continue

#         closed_set.add(key)

#         # now, generate the possible moves for the state that was just evaluated
#         valid, i, j = generate_possible_moves(state)

#         # run next round
#         if "up" in valid:
#             child = move_up(state, i, j).reshape(-1)
#             child_t = tuple(child)
#             if child_t not in parents:
#                 parents[child_t] = (tuple(state), node_index, curr_idx)
#                 open_list.append((child, node_index))
#                 node_index += 1
                

#         if "down" in valid:
#             child = move_down(state, i, j).reshape(-1)

#             child_t = tuple(child)
#             if child_t not in parents:
#                 parents[child_t] = (tuple(state), node_index, curr_idx)
#                 open_list.append((child, node_index))
#                 node_index += 1

#         if "right" in valid:
#             child = move_right(state, i, j).reshape(-1)

#             child_t = tuple(child)
#             if child_t not in parents:
#                 parents[child_t] = (tuple(state), node_index, curr_idx)
#                 open_list.append((child, node_index))
#                 node_index += 1

#         if "left" in valid:
#             child = move_left(state, i, j).reshape(-1)

#             child_t = tuple(child)
#             if child_t not in parents:
#                 parents[child_t] = (tuple(state), node_index, curr_idx)
#                 open_list.append((child, node_index))
#                 node_index += 1

# #----------Define Function to Execute BFS----------------------
# def breadth_first_search(start_state, goal_state):
#     """
#     Initialize the BFS by adding the start state to the open list and then 
#     calling the function to compare against the goal state. The BFS will continue 
#     until the goal is reached, at which point the path to the goal will be generated.
#     """
#     global node_index

#     # initialize the parent dictionary with the start state (which has no parent)
#     child_t = tuple(start_state)
#     parents[child_t] = (None, node_index, None)

#     # add first state to the open list
#     open_list.append((start_state, node_index))
#     node_index += 1 

#     # start loop
#     compare_against_goal(goal_state)

# # -------------once the goal is reached, re-construct the path to get there-----------------------
# def generate_path(final_state):
#     """
#     Once the goal state is reached, re-create the file path
#     using the Parents dictionary
#     """
#     print("Generating Path to Goal...")
#     print(final_state)
#     child = final_state

#     order = [tuple(child)]

#     # reconstruct the path by going through the parent dictionary until we reach the start state (which has no parent)
#     while parents[tuple(child)][0] is not None:
#         child = parents[tuple(child)][0] 
#         order.append(child)

#     # reverse the order
#     order.reverse()

#     # Transpose the output to match the style used in the slides
#     for i in range(len(order)):
#         order[i] = np.asarray(order[i]).reshape(3, 3)
#         order[i] = np.transpose(order[i])

#     generate_text_files(order)
    
# #----------Define Function to generate the desired text files----------------------
# def generate_text_files(order):
#     filename_1 = "Nodes.txt"
#     explored_states = parents.keys()
#     # (par_t, child_idx, par_idx) = parents.values()
#     try:
#         with open(filename_1, 'w') as file:
#             for row in explored_states:
#                 file.write(' '.join(map(str, row)) + '\n')

#     except IOError as e:
#         print(f"Error writing to file: {e}")
#     finally:
#         file.close()

#     filename_2 = "NodesInfo.txt"    
#     try:
#         with open(filename_2, 'w') as file:
#             file.write("Node Index | Parent Node Index | Node\n")
#             file.write("--------------------------------------\n")

#             # parents: child_t -> (par_t, child_idx, par_idx)
#             rows = []
#             for child_t, (par_t, child_idx, par_idx) in parents.items():
#                 file.write(f"{child_idx} \t {par_idx} \t" + ' '.join(map(str, child_t)) + '\n')
            
#     except IOError as e:
#         print(f"Error writing to file: {e}")
#     finally:
#         file.close()


#     filename_3 = "nodePath.txt"
#     for i in range(len(order)):
#         order[i] = np.asarray(order[i]).reshape(-1)

#     try:
#         with open(filename_3, 'w') as file:
#             # parents: child_t -> (par_t, child_idx, par_idx)
#             for state in order:                      # order = path from start → goal
#                 flat = np.asarray(state).reshape(-1) # 1×9
#                 file.write(" ".join(map(str, flat)) + "\n")
            
#     except IOError as e:
#         print(f"Error writing to file: {e}")
#     finally:
#         file.close()



# #----------Run BFS----------------------
# if __name__ == "__main__":
    # start_state = np.array([8, 6, 7, 2, 5, 4, 3, 0, 1])
    # goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])

    # breadth_first_search(start_state, goal_state)