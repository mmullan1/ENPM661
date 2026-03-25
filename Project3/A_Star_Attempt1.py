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
def move(child_t, c2c, ori, step_size, delta_theta):
    new_ori = ori + delta_theta
    theta = np.deg2rad(new_ori)

    new_x = child_t[0] + step_size * np.cos(theta)
    new_y = child_t[1] + step_size * np.sin(theta)

    new_state = (new_x, new_y, new_ori)
    new_cost = c2c + step_size

    return new_state, new_cost


def move_fwd(child_t, c2c, ori, step_size):
    return move(child_t, c2c, ori, step_size, 0)

def move_fwd_ccw30(child_t, c2c, ori, step_size):
    return move(child_t, c2c, ori, step_size, 30)

def move_fwd_ccw60(child_t, c2c, ori, step_size):
    return move(child_t, c2c, ori, step_size, 60)

def move_fwd_cw30(child_t, c2c, ori, step_size):
    return move(child_t, c2c, ori, step_size, -30)

def move_fwd_cw60(child_t, c2c, ori, step_size):
    return move(child_t, c2c, ori, step_size, -60)


#-----------------------------------------------------------
def run_AStar(start_pos, goal_pos, all_collisions, step_size):
    """
    Initialize the alg by adding the start state to the open list and then 
    calling the function to compare against the goal state. A* will continue 
    until the goal is reached, at which point the path to the goal will be generated.
    """
    t_init = time.perf_counter()
    global open_list

    # initialize the parent dictionary with the discretized start state (which has no parent)
    child_t = tuple(goal_pos)
    disc_child = discretize(child_t)
    c2c = 0
    ct = c2c + np.sqrt((start_pos[0]-goal_pos[0])**2 + (start_pos[1]-goal_pos[1])**2)

    open_cost[disc_child] = (None, c2c, child_t)
    heapq.heappush(open_list, (ct, child_t))

    # start loop
    t_fin = compare_against_goal(goal_pos, start_pos, all_collisions, step_size)

    return t_init, t_fin

#-----------------------------------------------------------
def compare_against_goal(goal_pos, start_pos, all_collisions, step_size):
    """
    The loop that runs until the open list is empty
    """
    while open_list:
        # pop cheapest node
        ct, child_t = heapq.heappop(open_list)
        disc_child = discretize(child_t)

        # skip stale heap entries
        if disc_child not in open_cost:
            continue

        parent, c2c, best_state = open_cost[disc_child]

        if child_t != best_state:
            continue

        # goal threshold check
        if (child_t[0] - start_pos[0])**2 + (child_t[1] - start_pos[1])**2 <= 1.5**2:
            print(f"GOAL REACHED; COST IS {c2c}")
            t_fin = time.perf_counter()
            order = generate_path(goal_pos, start_pos, child_t)
            animate_search_and_path(order, explored_nodes)
            return t_fin

        # skip if already closed
        if disc_child in closed_list:
            continue

        # close and expand
        closed_list.add(disc_child)
        explored_nodes.append(child_t)
        ori = child_t[2]

        generate_possible_moves(c2c, child_t, all_collisions, ori, step_size)

#-----------------------------------------------------------
def generate_possible_moves(c2c, child_t, all_collisions, ori, step_size):
    """
    From the previous location, generate possible next steps,
    evaluate to ensure they aren't in a collision or out of
    bounds, and then, if the entry isn't in open_cost or
    the cost is less than the entry in open_cost, add that entry
    to open_list
    """
    global open_list

    # define action obtions
    Actions_Set = {
        move_fwd, 
        move_fwd_ccw30,
        move_fwd_ccw60,
        move_fwd_cw30,
        move_fwd_cw60
    }

    # run through all possible actions
    for action in Actions_Set:
        new_state, new_cost = action(child_t, c2c, ori, step_size)
        x, y = new_state[0], new_state[1]

        # round to int
        xi = int(np.round(x))
        yi = int(np.round(y))

        # duplicate check
        disc_new = discretize(new_state)
        if disc_new in closed_list:
            continue

        ct = new_cost + np.sqrt((start_pos[0]-x)**2 + (start_pos[1]-y)**2)



        # check boundaries (0<x<600, 0<y<250)
        if yi < 0 or yi >= 250 or xi < 0 or xi >= 600:
            continue
        
        # check to see if it is in a collision
        if all_collisions[yi, xi]:
            continue

        # check to see if new_state has already been searched or if the new c2c for
        # new_cost is cheaper than the previously logged one; if either of these
        # conitions are met, add new_state to the open_list
        disc_child = discretize(child_t)
        disc_new = discretize(new_state)

        if disc_new not in open_cost or open_cost[disc_new][1] > new_cost:
            open_cost[disc_new] = (disc_child, new_cost, new_state)
            heapq.heappush(open_list, (ct, new_state))

#-----------------------------------------------------------
def discretize(node):
    x, y, theta = node

    ix = int(np.floor(x / 0.5 + 0.5))   # nearest 0.5 bin
    iy = int(np.floor(y / 0.5 + 0.5))
    itheta = int((theta % 360) // 30)   # 0..11

    return (ix, iy, itheta)
#-----------------------------------------------------------
def generate_path(goal_pos, start_pos, final_node):
    """
    Once the search is done, find the path from the start to the goal
    """

    current = discretize(final_node)
    order = [final_node]

    while current != discretize(goal_pos):
        parent, _, actual_state = open_cost[current]
        current = parent

        if current is not None:
            order.append(open_cost[current][2])

    # order.reverse()
    return order

#-----------------------------------------------------------
def animate_search_and_path(order, explored_nodes):
    plt.ion()

    ax = plt.gca()

    # --- Animate search tree ---
    for k, node in enumerate(explored_nodes):
        disc_node = discretize(node)

        if disc_node in open_cost:
            parent, _, _ = open_cost[disc_node]

            if parent is not None:
                parent_state = open_cost[parent][2]

                dx = node[0] - parent_state[0]
                dy = node[1] - parent_state[1]

                ax.quiver(parent_state[0], parent_state[1],
                          dx, dy,
                          angles='xy', scale_units='xy', scale=1,
                          color='b', width=0.002)

        if k % 100 == 0 or k == len(explored_nodes) - 1:
            plt.draw()
            plt.pause(0.01)

    # --- Animate final path ---
    for i in range(1, len(order)):
        x0, y0 = order[i-1][0], order[i-1][1]
        x1, y1 = order[i][0], order[i][1]

        dx = x1 - x0
        dy = y1 - y0

        ax.quiver(x0, y0,
                  dx, dy,
                  angles='xy', scale_units='xy', scale=1,
                  color='r', width=0.004)

        plt.draw()
        plt.pause(0.02)
#-----------------------------------------------------------
def draw_obstacle_course(order):
    """
    Construct a 600x250 MatPlotLib graph with the obstacle "MM 2577" (green)
    and add a boundary of 5 extra mm around that (yellow), accounting for the robot not being
    a point robot but a robot with a 5 mm radius
    """
    # Grid
    x = np.arange(0, 600)
    y = np.arange(0, 250)
    X, Y = np.meshgrid(x, y)

    X_plot, Y_plot = X, Y

    # Thickness parameter
    eps = 1

    # resample to fit (600, 250) grid instead of a (180, 50) grid
    SX = 600 / 180
    SY = 250 / 50

    X = X / SX
    Y = Y / SY

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
    def get_outer_ring(mask, r=5):
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
    E1_barrier = get_outer_ring(E1, r=5)
    E2_barrier = get_outer_ring(E2, r=5)
    E3_barrier = get_outer_ring(E3, r=5)
    E4_barrier = get_outer_ring(E4, r=5)
    E5_barrier = get_outer_ring(E5, r=5)
    E6_barrier = get_outer_ring(E6, r=5)

    all_obstacles = E1 | E2 | E3 | E4 | E5 | E6
    all_barriers  = E1_barrier | E2_barrier | E3_barrier | E4_barrier | E5_barrier | E6_barrier

    all_collisions = all_obstacles | all_barriers

    # initialize the plot the first time this fcn is called,
    # but not the second time
    if order is None:
        plt.ion()
        plt.figure(figsize=(6,6))
        plt.contourf(X_plot, Y_plot, all_obstacles, levels=[0.5, 1])
        plt.contourf(X_plot, Y_plot, all_barriers, levels=[0.5, 1], colors=['yellow'])
        plt.gca().set_aspect('equal')
        plt.xlim(0, 600)
        plt.ylim(0, 250)
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
        start_str = input("Choose starting x,y,theta coordinates: ")
        goal_str = input("Choose goal x,y,theta coordinates: ")

        # split into the x and y coords
        try:
            sx, sy, sd = map(int, start_str.split(","))
            gx, gy, gd = map(int, goal_str.split(","))

            start_pos = (sx, sy, sd)
            goal_pos = (gx, gy, gd)

        except:
            print("Invalid input. Use format: x,y")
            continue

        # boundary check
        if not (0 <= sx < 600 and 0 <= sy < 250):
            print("Start node outside map.")
            continue

        if not (0 <= gx < 600 and 0 <= gy < 250):
            print("Goal node outside map.")
            continue

        # collision check
        if all_collisions[sy, sx]:
            print("Start node is in obstacle space.")
            continue

        if all_collisions[gy, gx]:
            print("Goal node is in obstacle space.")
            continue

        # angle check (increments of 30 deg)
        if sd % 30 != 0:
            print("Start orientation is not in increments of 30 deg")
            continue

        if gd % 30 != 0:
            print("Goal orientation is not in increments of 30 deg")
            continue

        return start_pos, goal_pos
#----------Run dijkstra----------------------
if __name__ == "__main__":


    # generate all collision spaces
    order = None
    all_collisions = draw_obstacle_course(order)

    # Define the steps of each iteration
    step_size = 10

    # grab inputs from terminal entries
    start_pos, goal_pos = get_inputs(all_collisions)

    # run algorithm
    t_init, t_fin = run_AStar(start_pos, goal_pos, all_collisions, step_size)

    # evaluate time for algorithm to run
    # dt = t_fin - t_init
    # print(f"The algorithm takes {dt} s to run")

    # keep plot open after completion
    plt.show()