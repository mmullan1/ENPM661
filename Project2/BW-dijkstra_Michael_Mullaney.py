import heapq
import matplotlib.pyplot as plt
import numpy as np
import time

#-----------------------------------------------------------
# Global Vars
explored_nodes = []
open_cost = {}
open_list = []
closed_list = set()

#-----------------------------------------------------------
# Different Action Primitives
def move_right(child_t, c2c):
    new_state = (child_t[0] + 1, child_t[1])
    new_cost = c2c + 1
    return new_state, new_cost

def move_left(child_t, c2c):
    new_state = (child_t[0] - 1, child_t[1])
    new_cost = c2c + 1
    return new_state, new_cost

def move_up(child_t, c2c):
    new_state = (child_t[0], child_t[1] + 1)
    new_cost = c2c + 1
    return new_state, new_cost

def move_down(child_t, c2c):
    new_state = (child_t[0], child_t[1] - 1)
    new_cost = c2c + 1
    return new_state, new_cost

def move_up_right(child_t, c2c):
    new_state = (child_t[0] + 1, child_t[1] + 1)
    new_cost = c2c + 1.4
    return new_state, new_cost

def move_up_left(child_t, c2c):
    new_state = (child_t[0] - 1, child_t[1] + 1)
    new_cost = c2c + 1.4
    return new_state, new_cost

def move_down_right(child_t, c2c):
    new_state = (child_t[0] + 1, child_t[1] - 1)
    new_cost = c2c + 1.4
    return new_state, new_cost

def move_down_left(child_t, c2c):
    new_state = (child_t[0] - 1, child_t[1] - 1)
    new_cost = c2c + 1.4
    return new_state, new_cost

#-----------------------------------------------------------
def run_djistra(start_pos, goal_pos, all_collisions):
    """
    Initialize the alg by adding the start state to the open list and then 
    calling the function to compare against the goal state. djistra will continue 
    until the goal is reached, at which point the path to the goal will be generated.
    """
    t_init = time.perf_counter()
    global open_list

    # overlay start and goal nodes
    plt.plot(start_pos[0], start_pos[1], 'o', color="orange", label = "start", zorder = 10)
    plt.plot(goal_pos[0], goal_pos[1], '*', color="lime", label = "goal", zorder = 10)
    plt.legend(loc="upper left", bbox_to_anchor=(0.8, 1.5))
    # initialize the parent dictionary with the start state (which has no parent)
    child_t = tuple(goal_pos)
    c2c = 0

    # add cost of first state to open_cost dictionary; also store the parent
    open_cost[child_t] = (None, c2c)

    # add first state to the open list
    heapq.heappush(open_list, (c2c, child_t))

    # start loop
    t_fin = compare_against_goal(goal_pos, start_pos, all_collisions)

    return t_init, t_fin

#-----------------------------------------------------------
def compare_against_goal(goal_pos, start_pos, all_collisions):
    """
    The loop that runs until the open list is empty
    """
    while open_list:
        # remove the cheapest cost entiry from opon_list
        c2c, child_t = heapq.heappop(open_list)

        # if the cost is more expensive than another cost
        # associated with child_t, neglect it
        if c2c > open_cost[child_t][1]:
            continue

        # Check to see if the start_pos has been reached
        if child_t == start_pos:
            print(f"GOAL REACHED; COST IS {c2c}")
            t_fin = time.perf_counter()
            order = generate_path(goal_pos, start_pos, open_cost.get(child_t))
            animate_search_and_path(order, explored_nodes)
            return t_fin
        
        # add the child to the closed list if it's not already there
        # and generate the next movements
        if child_t not in closed_list:
            closed_list.add(child_t)
            explored_nodes.append(child_t)

            generate_possible_moves(c2c, child_t, all_collisions)

   
#-----------------------------------------------------------
def generate_possible_moves(c2c, child_t,  all_collisions):
    """
    From the previous location, generate possible next steps,
    evaluate to ensure they aren't in a collision or out of
    bounds, and then, if the entry isn't in open_cost or
    the cost is less than the entry in open_cost, add that entry
    to open_list
    """
    global open_list

    # define action obtions
    Actions_Set = (
        move_right,
        move_left,
        move_up,
        move_down,
        move_up_right,
        move_up_left,
        move_down_right,
        move_down_left
    )

    # run through all possible actions
    for action in Actions_Set:
        new_state, new_cost = action(child_t, c2c)
        x, y = new_state

        # check boundaries (0<x<180, 0<y<50)
        if y < 0 or y >= 50 or x < 0 or x >= 180:
            continue

        # check to see if it is in a collision
        if all_collisions[y, x]:
            continue

        # check to see if new_state has already been searched or if the new c2c for
        # new_cost is cheaper than the previously logged one; if either of these
        # conitions are met, add new_state to the open_list
        if new_state not in open_cost or open_cost.get(new_state)[1] > (new_cost):
            open_cost[new_state] = (child_t, new_cost)

            heapq.heappush(open_list, (new_cost, new_state))

#-----------------------------------------------------------
def generate_path(goal_pos, start_pos, final_loc):
    """
    Once the search is done, find the path from the start to the goal
    """

    # extract the parent from the final_loc
    parent, cost = final_loc

    # add the final_loc to the ordered list
    order = [tuple(start_pos)]
    order.append(parent)

    # backtrack to find path from start to goal
    while parent != goal_pos: 
        parent, dum = open_cost.get(parent)
        order.append((parent))
    
    # Initialize the animation
    draw_obstacle_course(order)
    
    return order

#-----------------------------------------------------------
def animate_search_and_path(order, explored_nodes):
    """
    Once the search is done, animate the search on the MatPlotLib figure
    """

    plt.ioff()
    
    # animate explored nodes
    ex = []
    ey = []
    for k, node in enumerate(explored_nodes):
        ex.append(node[0])
        ey.append(node[1])

        if k % 10 == 0 or k == len(explored_nodes) - 1:
            plt.plot(ex, ey, 'b.', markersize=2)
            plt.draw()
            plt.pause(0.01)
            ex.clear()
            ey.clear()

    # add final path
    for i in range(1, len(order)):
        plt.plot([order[i-1][0], order[i][0]],
                [order[i-1][1], order[i][1]],
                'r-', linewidth=2)

        plt.draw()
        plt.pause(0.02)

#-----------------------------------------------------------
def draw_obstacle_course(order):
    """
    Construct a 180x50 MatPlotLib graph with the obstacle "MM 2577" (green)
    and add a boundary of 2 extra mm around that (yellow)
    """
    # Grid
    x = np.arange(0, 180)
    y = np.arange(0, 50)
    X, Y = np.meshgrid(x, y)

    # Thickness parameter
    eps = 1

    # --- Define the First Entry M as a semi-algebraic set ---
    left_bar = (X - 5)**2 <= eps**2
    left_bar &= (Y >= 10) & (Y <= 40)

    right_bar = (X - 25)**2 <= eps**2
    right_bar &= (Y >= 10) & (Y <= 40)

    left_diag = (Y - (-2*(X - 5) + 40))**2 <= eps**2
    left_diag &= (X >= 5) & (X <= 15)
    left_diag &= (Y >= 20) & (Y <= 40)


    right_diag = (Y - (2*(X - 25) + 40))**2 <= eps**2
    right_diag &= (X >= 15) & (X <= 25)
    right_diag &= (Y >= 20) & (Y <= 40)

    E1 = left_bar | right_bar | left_diag | right_diag

    # --- Define the Second Entry M as a semi-algebraic set ---
    left_bar2 = ((X-30) - 5)**2 <= eps**2
    left_bar2 &= (Y >= 10) & (Y <= 40)

    right_bar2 = ((X-30) - 25)**2 <= eps**2
    right_bar2 &= (Y >= 10) & (Y <= 40)

    left_diag2 = (Y - (-2*((X-30) - 5) + 40))**2 <= eps**2
    left_diag2 &= ((X-30) >= 5) & ((X-30) <= 15)
    left_diag2 &= (Y >= 20) & (Y <= 40)

    right_diag2 = (Y - (2*((X-30) - 25) + 40))**2 <= eps**2
    right_diag2 &= ((X-30) >= 15) & ((X-30) <= 25)
    right_diag2 &= (Y >= 20) & (Y <= 40)

    E2 = left_bar2 | right_bar2 | left_diag2 | right_diag2


    # --- Define the Third Entry 2 as a semi-algebraic set ---
    top = ((X-75)**2 + (Y-32)**2 >= (9-eps/2)**2)
    top &= ((X-75)**2 + (Y-32)**2 <= (9+eps/2)**2)
    top &= (X >= 66) & (X <= 84)
    top &= (Y >= 32)

    diag = (Y - (1.11*(X-84) + 32))**2 <= eps**2
    diag &= (X >= 66) & (X <= 84)
    diag &= (Y >= 12) & (Y <= 32)

    bottom = (X >= 66) & (X <= 84)
    bottom &= (Y - 12)**2 <= eps**2

    E3 = top | diag | bottom
        
    # --- Define the Fourth Entry 5 as a semi-algebraic set ---
    top = (X >= 93) & (X <= 108)
    top &= (Y - 40)**2 <= eps**2

    mid = (X - 93)**2 <= eps**2
    mid &= (Y >= 30) & (Y <= 40)

    bottom = (((X-3)-93)**2 + (Y-20)**2 >= (10-eps/1.5)**2)
    bottom &= (((X-3)-93)**2 + (Y-20)**2 <= (10+eps/1.5)**2)
    bottom &= (X >= 93) 
    bottom &= (Y >= 10) & (Y <= 35)

    E4 = top | mid | bottom

    # --- Define the Fifth Entry 7 as a semi-algebraic set ---
    top = (X >= 115) & (X <= 135)
    top &= (Y - 40)**2 <= eps**2

    right_diag = (Y - (2*(X - 135) + 40))**2 <= 2*eps**2
    right_diag &= (X >= 115) & (X <= 135)
    right_diag &= (Y >= 10) & (Y <= 40)

    E5 = top | right_diag

    # --- Define the Sixth Entry 7 as a semi-algebraic set ---
    top = (X >= 145) & (X <= 165)
    top &= (Y - 40)**2 <= eps**2

    right_diag = (Y - (2*(X - 165) + 40))**2 <= 2*eps**2
    right_diag &= (X >= 145) & (X <= 165)
    right_diag &= (Y >= 10) & (Y <= 40)

    E6 = top | right_diag

    # --- Define fcn to generate boundary  ---
    def get_outer_ring(mask, r=2):
        """
        Generate collision boundary
        """
        expanded = np.zeros_like(mask, dtype=bool)

        rows, cols = mask.shape

        for di in range(-r, r+1):
            for dj in range(-r, r+1):
                src_r0 = max(0, -di)
                src_r1 = min(rows, rows - di)
                src_c0 = max(0, -dj)
                src_c1 = min(cols, cols - dj)

                dst_r0 = max(0, di)
                dst_r1 = min(rows, rows + di)
                dst_c0 = max(0, dj)
                dst_c1 = min(cols, cols + dj)

                expanded[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]

        outer_ring = expanded & (~mask)
        return outer_ring
    
    # --- Generate boundaries  ---
    E1_barrier = get_outer_ring(E1, r=2)
    E2_barrier = get_outer_ring(E2, r=2)
    E3_barrier = get_outer_ring(E3, r=2)
    E4_barrier = get_outer_ring(E4, r=2)
    E5_barrier = get_outer_ring(E5, r=2)
    E6_barrier = get_outer_ring(E6, r=2)

    all_obstacles = E1 | E2 | E3 | E4 | E5 | E6
    all_barriers  = E1_barrier | E2_barrier | E3_barrier | E4_barrier | E5_barrier | E6_barrier

    all_collisions = all_obstacles | all_barriers

    # initialize the plot the first time this fcn is called,
    # but not the second time
    if order is None:
        plt.ion()
        plt.figure(figsize=(6,6))
        plt.contourf(X, Y, all_obstacles, levels=[0.5, 1])
        plt.contourf(X, Y, all_barriers, levels=[0.5, 1], colors=['yellow'])
        plt.gca().set_aspect('equal')
        plt.xlim(0, 180)
        plt.ylim(0, 50)
        plt.grid(True)
        plt.title("MM 2577 as a Semi-Algebraic Set")
        

    return all_collisions

#-----------------------------------------------------------
def get_inputs(all_collisions):
    """
    Have the user input the start and goal locations, and have them redo if if their
    entries are in collision boundaries
    """
    while True:

        # have the user input the start and goal positions
        start_str = input("Choose starting x,y coordinates: ")
        goal_str = input("Choose goal x,y coordinates: ")

        # split into the x and y coords
        try:
            sx, sy = map(int, start_str.split(","))
            gx, gy = map(int, goal_str.split(","))

            start_pos = (sx, sy)
            goal_pos = (gx, gy)

        except:
            print("Invalid input. Use format: x,y")
            continue

        # boundary check
        if not (0 <= sx < 180 and 0 <= sy < 50):
            print("Start node outside map.")
            continue

        if not (0 <= gx < 180 and 0 <= gy < 50):
            print("Goal node outside map.")
            continue

        # collision check
        if all_collisions[sy, sx]:
            print("Start node is in obstacle space.")
            continue

        if all_collisions[gy, gx]:
            print("Goal node is in obstacle space.")
            continue

        return start_pos, goal_pos
#----------Run dijkstra----------------------
if __name__ == "__main__":


    # generate all collision spaces
    order = None
    all_collisions = draw_obstacle_course(order)

    # grab inputs from terminal entries
    start_pos, goal_pos = get_inputs(all_collisions)

    # run algorithm
    t_init, t_fin = run_djistra(start_pos, goal_pos, all_collisions)

    # evaluate time for algorithm to run
    dt = t_fin - t_init
    print(f"The algorithm takes {dt} s to run")

    # keep plot open after completion
    plt.show()