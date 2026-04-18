import heapq
import matplotlib.pyplot as plt
import numpy as np
import time
import math

#-----------------------------------------------------------
# Global Vars
explored_nodes = []
open_cost = {}
open_list = []
closed_list = set()

#-----------------------------------------------------------
# reset each time
def reset_search_state():
    global explored_nodes, open_cost, open_list, closed_list
    explored_nodes = []
    open_cost = {}
    open_list = []
    closed_list = set()

#-----------------------------------------------------------
# Differential Drive Constraints and Action Primitives
def cost(Xi, Yi, Thetai, UL, UR, all_collisions):
    t = 0
    r = 3.3      # cm
    L = 28.7     # cm
    dt = 0.1

    # convert rpm -> rad/s
    UL = UL * 2 * math.pi / 60
    UR = UR * 2 * math.pi / 60

    Xn = Xi
    Yn = Yi
    Thetan = math.radians(Thetai)

    D = 0
    while t < 0.5:
        t += dt

        vx = 0.5 * r * (UL + UR)
        w = (r / L) * (UR - UL)

        Delta_Xn = vx * math.cos(Thetan) * dt
        Delta_Yn = vx * math.sin(Thetan) * dt

        Xn += Delta_Xn
        Yn += Delta_Yn
        Thetan += w * dt

        D += math.sqrt(Delta_Xn**2 + Delta_Yn**2)


        # check intermediate collision
        x_d, y_d, _ = grid_index((Xn, Yn, math.degrees(Thetan)))

        if x_d < 0 or x_d >= all_collisions.shape[1] or y_d < 0 or y_d >= all_collisions.shape[0]:
            return None

        if all_collisions[y_d, x_d]:
            return None

    Thetan = math.degrees(Thetan)
    return Xn, Yn, Thetan, D
    
    
 #-----------------------------------------------------------   
def run_actions(Xi, Yi, Thetai, RPM1, RPM2, all_collisions):
    actions=[[0, RPM1], [RPM1, 0],[RPM1, RPM1],[0, RPM2],[RPM2, 0],[RPM2, RPM2], [RPM1, RPM2], [RPM2, RPM1]]
    actions_set = []
    for action in actions:
        k=cost(Xi,Yi, Thetai, action[0], action[1], all_collisions) 
        # print(k)
        actions_set.append(k)
    return actions_set


#-----------------------------------------------------------
def grid_index(node):

    x, y, theta = node

    ix = int(np.floor(x / 0.5 + 0.5))   # nearest 0.5 bin
    iy = int(np.floor(y / 0.5 + 0.5))
    itheta = int((theta % 360) // 30)   # 0.11

    return (ix, iy, itheta)

#-----------------------------------------------------------
def discretize(node):

    x, y, theta = node

    ix = int(np.floor(x / 5 + 0.5))   # nearest 2 bin
    iy = int(np.floor(y / 5 + 0.5))
    itheta = int((theta % 360) // 30)   # 0.11

    return (ix, iy, itheta)

#-----------------------------------------------------------
def run_AStar(start_pos, goal_pos, all_collisions, clearance, RPM1, RPM2):
    """
    Initialize the alg by adding the start state to the open list and then 
    calling the function to compare against the goal state. A* will continue 
    until the goal is reached, at which point the path to the goal will be generated.
    """
    global t_init
    global open_list
    reset_search_state()
    t_init = time.perf_counter()



    plt.plot(start_pos[0], start_pos[1], 'o', color="orange", label = "start", zorder = 10)
    plt.plot(goal_pos[0], goal_pos[1], '*', color="lime", label = "goal", zorder = 10)
    plt.legend(loc="upper left", bbox_to_anchor=(0.8, 1.5))

    # initialize the parent dictionary with the start state (which has no parent)
    child_t = tuple(start_pos)
    # print(f"CHILD_T: {child_t}")

    # discretize child_t
    child_t_disc = discretize(child_t)
    info = (child_t_disc, child_t)

    # initialize cost to come and total cost
    c2c = 0
    ct = c2c + np.sqrt((start_pos[0]-goal_pos[0])**2 + (start_pos[1]-goal_pos[1])**2)

    # add to dictionary
    open_cost[child_t_disc] = (None, c2c, start_pos)

    # add to heap
    heapq.heappush(open_list, (ct, info))

    # start loop
    t_fin = compare_against_goal(goal_pos, start_pos, all_collisions, clearance, RPM1, RPM2)

    return t_init, t_fin

#-----------------------------------------------------------
def compare_against_goal(goal_pos, start_pos, all_collisions, clearance, RPM1, RPM2):
    """
    The loop that runs until the open list is empty
    """
    while open_list:
        # remove the cheapest cost entiry from opon_list
        ct, (child_t_disc, child_t) = heapq.heappop(open_list)

        # extract info from dictionary
        parent_disc_state, c2c, actual_state = open_cost[child_t_disc]
        # actual_state = tuple(map(int, actual_state))


        # Check to see if the start_pos has been reached
        if (actual_state[0] - goal_pos[0])**2 + (actual_state[1] - goal_pos[1])**2 <= 5**2:
            print(f"GOAL REACHED; COST IS {c2c}")
            t_fin = time.perf_counter()
            order = generate_path(goal_pos, start_pos, open_cost.get(child_t_disc), clearance)
            animate_search_and_path(order, explored_nodes, t_fin)
            return t_fin
        
        # add the child to the closed list if it's not already there
        # and generate the next movements
        if child_t_disc not in closed_list:
            closed_list.add(child_t_disc)
            explored_nodes.append(child_t)

            generate_possible_moves(c2c, actual_state, goal_pos, all_collisions, RPM1, RPM2)

    # return t_fin
#-----------------------------------------------------------
def generate_possible_moves(c2c, actual_state, goal_pos, all_collisions, RPM1, RPM2):
    """
    From the previous location, generate possible next steps,
    evaluate to ensure they aren't in a collision or out of
    bounds, and then, if the entry isn't in open_cost or
    the cost is less than the entry in open_cost, add that entry
    to open_list
    """
    global open_list
    # print(actual_state)
    Actions_Set = run_actions(actual_state[0], actual_state[1], actual_state[2], RPM1, RPM2, all_collisions)
    # print(f"ACTIONS_SET: {Actions_Set}")

    # run through all possible actions
    for action in Actions_Set:
        # print(f"ACTION: {action}")
        # generate new position after action
        if action is not None:
            x, y, theta, D = action
            new_state = (x, y, theta)

            # discretize new position
            new_state_disc = discretize(new_state)
            x_d, y_d, theta_d = new_state_disc

            # update cost
            new_c2c = c2c + D
            ct = new_c2c + np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)

            new_state_disc_c = grid_index(new_state)
            x_c, y_c, theta_c = new_state_disc_c

            # check boundaries
            if x_c < 0 or x_c >= all_collisions.shape[1] or y_c < 0 or y_c >= all_collisions.shape[0]:
                continue

            if all_collisions[y_c, x_c]:
                continue

            # add to open list if cost is less than previous entry or if it isnt already in the open list
            if new_state_disc not in open_cost or open_cost.get(new_state_disc)[1] > new_c2c:
                open_cost[new_state_disc] = (actual_state, new_c2c, new_state)
                heapq.heappush(open_list, (ct, (new_state_disc, new_state)))

#-----------------------------------------------------------
def theta_equiv_dist(theta1, theta2):
    d = abs((theta1 - theta2) % 360)
    d = min(d, 360 - d)        # wrap to shortest angle
    return d

#-----------------------------------------------------------
def generate_path(goal_pos, start_pos, final_loc, clearance):
    parent, cost, child = final_loc

    order = [child]

    while parent is not None:
        order.append(parent)
        parent, cost, child = open_cost[discretize(parent)]

    draw_obstacle_course(order, clearance)
    return order[::-1]

#-----------------------------------------------------------
def animate_search_and_path(order, explored_nodes, t_fin):
    """
    Animate the explored search tree sparsely and the final path smoothly.

    Search animation:
        - Downsamples automatically
        - Draws at most 10,000 explored segments

    Final path animation:
        - Draws every segment
        - Uses line segments instead of quiver for smoother playback
    """
    global t_init

    plt.ion()
    ax = plt.gca()

    # --- Timing ---
    dt = t_fin - t_init
    print(f"The algorithm takes {dt:.3f} s to run")
    print(f"EXPLORED NODES: {len(explored_nodes)}")

    # =========================================================
    # 1) Animate explored search tree (SPARSE)
    # =========================================================
    max_search_segments = 1000000
    step = max(1, len(explored_nodes) // max_search_segments)

    # batch size for plotting several sampled segments at once
    search_batch = 50
    sx, sy = [], []

    sampled_indices = range(0, len(explored_nodes), step)

    for count, k in enumerate(sampled_indices):
        node = explored_nodes[k]
        disc_node = discretize(node)

        if disc_node in open_cost:
            parent, _, _ = open_cost[disc_node]

            if parent is not None:
                parent_disc = discretize(parent)

                if parent_disc in open_cost:
                    parent_state = open_cost[parent_disc][2]

                    # store one line segment, with None separator
                    sx.extend([parent_state[0], node[0], None])
                    sy.extend([parent_state[1], node[1], None])

        # draw every batch
        if (count + 1) % search_batch == 0 or k >= len(explored_nodes) - step:
            if sx:
                ax.plot(sx, sy, 'b-', linewidth=0.5)
                plt.draw()
                plt.pause(0.001)
                sx.clear()
                sy.clear()

    # =========================================================
    # 2) Animate final path (FULL)
    # =========================================================
    order = order[::-1]   # safer than in-place reverse

    write_path(order)

    path_batch = 3
    px, py = [], []

    for i in range(1, len(order)):
        x0, y0 = order[i - 1][0], order[i - 1][1]
        x1, y1 = order[i][0], order[i][1]

        px.extend([x0, x1, None])
        py.extend([y0, y1, None])

        if i % path_batch == 0 or i == len(order) - 1:
            ax.plot(px, py, 'r-', linewidth=2)
            plt.draw()
            plt.pause(0.001)
            px.clear()
            py.clear()

    plt.draw()
#-----------------------------------------------------------
def draw_obstacle_course(order, clearance, start_pos=None, r_bot=None):
    """
    Construct a 400x200 MatPlotLib graph with the obstacle course for
    Project 3 Phase 2 (green) and add a boundary of extra 
    clearance + robot radius (yellow).

    Optional:
        start_pos = (x, y, theta) or (x, y, None)
        r_bot     = robot radius in cm
    """
    # Grid
    x = np.arange(0, 200, 0.5)
    y = np.arange(0, 400, 0.5)
    X, Y = np.meshgrid(x, y)

    X_plot = X
    Y_plot = Y

    # --- Fcn for constructing half planes ---
    def in_half_plane(x, y, A, B, C):
        """
        Defines a closed half-plane Ax + By <= C
        """
        return A * x + B * y <= C

    # Cube 1
    cube1 = (
        in_half_plane(X, Y, -1, 0, -139.8) &
        in_half_plane(X, Y,  1, 0,  170.2) &
        in_half_plane(X, Y,  0,-1, -26.8) &
        in_half_plane(X, Y,  0, 1,  57.2)
    )

    cube2 = (
        in_half_plane(X, Y, -1, 0, -10.8) &
        in_half_plane(X, Y,  1, 0,  41.2) &
        in_half_plane(X, Y,  0,-1, -204.8) &
        in_half_plane(X, Y,  0, 1,  235.2)
    )

    cube3 = (
        in_half_plane(X, Y, -1, 0, -29.8) &
        in_half_plane(X, Y,  1, 0,  60.2) &
        in_half_plane(X, Y,  0,-1, -118.3) &
        in_half_plane(X, Y,  0, 1,  148.7)
    )

    wall1 = (
        in_half_plane(X, Y, -1, 0, 0) &
        in_half_plane(X, Y,  1, 0, 145) &
        in_half_plane(X, Y,  0,-1, -290) &
        in_half_plane(X, Y,  0, 1, 295)
    )

    theta = np.deg2rad(30)
    x0 = 0
    y0 = 38.4
    L1 = 140.0
    t = 5.0

    ux, uy = np.cos(theta), np.sin(theta)
    nx, ny = -np.sin(theta), np.cos(theta)

    wall2 = (
        in_half_plane(X - x0, Y - y0,  ux,  uy,  L1) &
        in_half_plane(X - x0, Y - y0, -ux, -uy,  0) &
        in_half_plane(X - x0, Y - y0,  nx,  ny,  t/2) &
        in_half_plane(X - x0, Y - y0, -nx, -ny,  t/2)
    )

    theta = np.deg2rad(30)
    x0 = 200.0
    y0 = 126.0
    L2 = 133.75
    t = 5.0

    ux, uy = -np.cos(theta), np.sin(theta)
    nx, ny = -uy, ux

    wall3 = (
        in_half_plane(X - x0, Y - y0,  ux,  uy,  L2) &
        in_half_plane(X - x0, Y - y0, -ux, -uy,  0) &
        in_half_plane(X - x0, Y - y0,  nx,  ny,  t/2) &
        in_half_plane(X - x0, Y - y0, -nx, -ny,  t/2)
    )

    all_obstacles = cube1 | cube2 | cube3 | wall1 | wall2 | wall3
    def get_outer_ring(mask, r=clearance):
            r = int(np.floor(r))
            expanded = np.zeros_like(mask, dtype=bool)

            rows, cols = mask.shape

            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):

                    # keep only points inside circular radius
                    if di**2 + dj**2 > r**2:
                        continue

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
    cube1_barrier = get_outer_ring(cube1, r=clearance)
    cube2_barrier = get_outer_ring(cube2, r=clearance)
    cube3_barrier = get_outer_ring(cube3, r=clearance)
    wall1_barrier = get_outer_ring(wall1, r=clearance)
    wall2_barrier = get_outer_ring(wall2, r=clearance)
    wall3_barrier = get_outer_ring(wall3, r=clearance)

    all_barriers = cube1_barrier | cube2_barrier | cube3_barrier | wall1_barrier | wall2_barrier | wall3_barrier
    all_collisions = all_obstacles | all_barriers

    if order is None:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 6))

        ax.contourf(X_plot, Y_plot, all_obstacles, levels=[0.5, 1])
        ax.contourf(X_plot, Y_plot, all_barriers, levels=[0.5, 1], colors=['yellow'])

        ax.set_aspect('equal')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 400)
        ax.grid(True)
        ax.set_title("Project 3 Phase 2 Obstacle Course")

    return all_collisions


#-----------------------------------------------------------
def get_inputs():
    """
    Have the user input the start and goal locations, and have them redo if if their
    entries are in collision boundaries
    """
    while True:

        # have the user input the start and goal positions
        start_str = input("Choose starting x,y,theta coordinates: ")
        goal_str = input("Choose goal x,y coordinates: ")
        clearance = input("Choose the clearance around the obstacles in mm:")
        rpm_str = input("Choose the rpm values (RPM1, RPM2) for the wheels:")
        # split into the x and y coords
        try:
            sx, sy, st = map(int, start_str.split(","))
            gx, gy = map(int, goal_str.split(","))

            start_pos = (sx, sy, st)
            goal_pos = (gx, gy, None)

            # convert clearance to cm and add robot radius
            clearance_mm = float(clearance)
            clearance_cm = 0.1*clearance_mm 

            rpm1, rpm2 = map(int, rpm_str.split(","))


        except:
            print("Invalid input. Use format: x,y")
            continue

        # boundary check
        if not (0 <= sx < 200 and 0 <= sy < 400):
            print("Start node outside map.")
            continue

        if not (0 <= gx < 200 and 0 <= gy < 400):
            print("Goal node outside map.")
            continue


        return start_pos, goal_pos, clearance_cm, rpm1, rpm2
    
#-----------------------------------------------------------
def check_collisions(all_collisions, start_pos, goal_pos, order, clearance_cells, RPM1, RPM2, r_bot):
    while True:
        sx, sy, _ = start_pos
        gx, gy, _ = goal_pos

        sx_d, sy_d = grid_index([sx, sy, 0])[:2]
        gx_d, gy_d = grid_index([gx, gy, 0])[:2]

        if all_collisions[sy_d, sx_d]:
            print("Start point is in collision")
            plt.close()

        elif all_collisions[gy_d, gx_d]:
            print("Goal point is in collision")
            plt.close()

        else:
            return start_pos, goal_pos, all_collisions, clearance_cells, RPM1, RPM2

        # only runs if there WAS a collision
        start_pos, goal_pos, clearance_cm, RPM1, RPM2 = get_inputs()

        # convert clearance to cm and add robot radius 
        clearance_total_cm = clearance_cm + r_bot
        clearance_cells = clearance_total_cm / 0.5


        all_collisions = draw_obstacle_course(order, clearance_cells)


#-----------------------------------------------------------
def write_path(order, points_per_meter=200, filename="astar_path.csv"):

    order = np.asarray(order, dtype=float)

    # keep x,y only
    xy_cm = order[:, 0:2]

    # convert cm -> m for ROS/Gazebo
    xy_m = xy_cm / 100.0

    # redistribute in meters
    new_path = redistribute_path(xy_m, points_per_meter)

    # save to CSV
    np.savetxt(filename, new_path, delimiter=",", header="x,y", comments="")

    print(f"Saved {len(new_path)} waypoints to {filename}")
    print(new_path)

    return new_path

#-----------------------------------------------------------
def redistribute_path(order, points_per_meter=200):
    """
    Redistribute a 2D path by arc length.

    Parameters
    ----------
    order : array-like, shape (N,2) or (N,>=2)
        Original path points. Only x and y are used.
    points_per_meter : float
        Desired waypoint density.

    Returns
    -------
    new_path : ndarray, shape (M,2)
        Redistributed path points.
    t_old : ndarray, shape (N,)
        Original normalized arc-length parameter in [0,1].
    t_new : ndarray, shape (M,)
        New uniformly spaced normalized parameter in [0,1].
    total_length : float
        Total path length in meters.
    """

    order = np.asarray(order, dtype=float)
    xy = order[:, 0:2]

    # Handle degenerate cases
    if len(xy) == 0:
        return np.empty((0, 2)), np.array([]), np.array([]), 0.0
    if len(xy) == 1:
        return xy.copy(), np.array([0.0]), np.array([0.0]), 0.0

    # Segment lengths
    diffs = np.diff(xy, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)

    # Cumulative arc length
    s = np.zeros(len(xy))
    s[1:] = np.cumsum(seg_lengths)

    total_length = s[-1]

    # If all points are identical
    if total_length == 0:
        return xy[:1].copy(), np.zeros(1), np.zeros(1), 0.0

    # Normalize to parameter t in [0,1]
    t_old = s / total_length

    # Number of redistributed points
    num_points = max(2, int(np.round(total_length * points_per_meter)) + 1)

    # Uniform parameter values in [0,1]
    t_new = np.linspace(0.0, 1.0, num_points)

    # Convert back to arc length values
    s_new = t_new * total_length

    # Interpolate x(s), y(s)
    x_new = np.interp(s_new, s, xy[:, 0])
    y_new = np.interp(s_new, s, xy[:, 1])

    new_path = np.column_stack((x_new, y_new))

    return new_path


#-----------------------------------------------------------
if __name__ == "__main__":


    # generate all collision spaces
    order = None
    # draw_obstacle_course(order, clearance=2)
    # get user inputs
    # define robot radius and apply scaling
    r_bot = 15 # cm

    start_pos, goal_pos, clearance_cm, RPM1, RPM2 = get_inputs()
    clearance_total_cm = clearance_cm + r_bot
    clearance_cells = clearance_total_cm / 0.5

    # construct obstacle course
    all_collisions = draw_obstacle_course(order, clearance_cells)

    # check for collisions
    start_pos, goal_pos, all_collisions, clearance, RPM1, RPM2 = check_collisions(all_collisions, start_pos, goal_pos, order, clearance_cells, RPM1, RPM2, r_bot)


    # run algorithm
    t_init, t_fin = run_AStar(start_pos, goal_pos, all_collisions, clearance, RPM1, RPM2)


    # # keep plot open after completion
    plt.ioff()
    plt.show(block=True)