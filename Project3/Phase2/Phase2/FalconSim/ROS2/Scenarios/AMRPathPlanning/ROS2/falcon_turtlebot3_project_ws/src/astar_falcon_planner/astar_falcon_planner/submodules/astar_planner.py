import heapq
import numpy as np
import time
import math

# -----------------------------------------------------------
# Global Vars
explored_nodes = []
open_cost = {}
open_list = []
closed_list = set()

# -----------------------------------------------------------
def reset_search_state():
    """
    ensures all variables are reset between consecutive runs of this script
    """
    global explored_nodes, open_cost, open_list, closed_list
    explored_nodes = []
    open_cost = {}
    open_list = []
    closed_list = set()

# -----------------------------------------------------------
# Differential Drive Constraints and Action Primitives
def cost(Xi, Yi, Thetai, UL, UR, all_collisions, delta_time, wheel_radius, wheel_distance):
    """
    Simulate one action primitive

    Inputs:
        Xi, Yi       : current position in cm
        Thetai       : current heading in radians
        UL, UR       : wheel speeds in RPM
        all_collisions : collision grid
        delta_time   : action duration in seconds
        wheel_radius : cm
        wheel_distance : cm

    Returns:
        (Xn, Yn, Thetan, D) or None
    """
    t = 0.0
    dt = 0.1

    r = wheel_radius
    L = wheel_distance

    # convert rpm -> rad/s
    UL = UL * 2.0 * math.pi / 60.0
    UR = UR * 2.0 * math.pi / 60.0

    Xn = Xi
    Yn = Yi
    Thetan = math.radians(Thetai)

    D = 0.0

    while t < delta_time:
        step_dt = min(dt, delta_time - t)
        t += step_dt

        vx = 0.5 * r * (UL + UR)
        w = (r / L) * (UR - UL)

        Delta_Xn = vx * math.cos(Thetan) * step_dt
        Delta_Yn = vx * math.sin(Thetan) * step_dt

        Xn += Delta_Xn
        Yn += Delta_Yn
        Thetan += w * step_dt

        D += math.sqrt(Delta_Xn**2 + Delta_Yn**2)

        x_d, y_d, _ = grid_index((Xn, Yn, math.degrees(Thetan)))

        if x_d < 0 or x_d >= all_collisions.shape[1] or y_d < 0 or y_d >= all_collisions.shape[0]:
            return None

        if all_collisions[y_d, x_d]:
            return None

    Thetan = math.degrees(Thetan)
    return Xn, Yn, Thetan, D

# -----------------------------------------------------------
def run_actions(Xi, Yi, Thetai, RPM1, RPM2, all_collisions, delta_time, wheel_radius, wheel_distance):
    """
    Computes all possible next actions for a given state using predefined
    differential drive wheel RPM combinations.

    Inputs:
        Xi, Yi           : current position in cm
        Thetai           : current heading in degrees
        RPM1, RPM2       : two discrete wheel speed values (RPM)
        all_collisions   : boolean collision map
        delta_time       : duration of each action (seconds)
        wheel_radius     : wheel radius in cm
        wheel_distance   : distance between wheels in cm

    Returns:
        actions_set : list of resulting states from each action,
                      where each entry is either:
                          (Xn, Yn, Thetan, D)
                      or None if the action results in collision or goes out of bounds
    """
    actions = [
        [0, RPM1], [RPM1, 0], [RPM1, RPM1],
        [0, RPM2], [RPM2, 0], [RPM2, RPM2],
        [RPM1, RPM2], [RPM2, RPM1]
    ]

    actions_set = []
    for action in actions:
        k = cost(
            Xi, Yi, Thetai,
            action[0], action[1],
            all_collisions,
            delta_time,
            wheel_radius,
            wheel_distance
        )
        actions_set.append(k)
    return actions_set

# -----------------------------------------------------------
def grid_index(node):
    """
    Discretizes the 200x400 cm grid into 0.5 cm and 30 degree bins
    
    Inputs:
        node: the input node that is placed in its nearest bin

    Returns:
        ix: the nearest x-bin to the input node
        iy: the nearest y-bin for the input node
        itheta: the nearest theta-bin for the input node
    """
    x, y, theta = node
    ix = int(np.floor(x / 0.5 + 0.5))      # nearest 0.5 cm bin
    iy = int(np.floor(y / 0.5 + 0.5))
    itheta = int((theta % 360) // 30)
    return (ix, iy, itheta)

# -----------------------------------------------------------
def discretize(node):
    """
    Discretizes the 200x400 cm grid into 5.0 cm and 30 degree bins.
    Used for path search to reduce resolution and improve computational speed.
    
    Inputs:
        node: the input node that is placed in its nearest bin

    Returns:
        ix: the nearest x-bin to the input node
        iy: the nearest y-bin for the input node
        itheta: the nearest theta-bin for the input node
    """
    x, y, theta = node
    ix = int(np.floor(x / 5.0 + 0.5))      # coarser search bin
    iy = int(np.floor(y / 5.0 + 0.5))
    itheta = int((theta % 360) // 30)
    return (ix, iy, itheta)

# -----------------------------------------------------------
def in_half_plane(x, y, A, B, C):
    """
    Check whether points (x, y) lie inside a half-plane.

    The half-plane is defined by the linear inequality:
        A*x + B*y <= C

    Inputs:
        x, y : coordinates (scalars or arrays)
        A, B : coefficients defining the line normal
        C    : threshold defining the boundary

    Returns:
        Boolean mask where True indicates the point lies
        inside or on the boundary of the half-plane.
    """
    return A * x + B * y <= C

# -----------------------------------------------------------
def build_collision_map(clearance):
    """
    Rebuilds the obstacle course inside astar_planner.py so Falcon can call it
    directly without needing the plotting script.
    Clearance is in cm.

    Inputs:
        clearance : the obstacle inflation radius in cm, used to expand each
                    obstacle outward and create a collision buffer

    Returns: 
        all_collisions: a boolean collision map over the 200x400 cm workspace,
                        where True indicates obstacle space or inflated
                        clearance region and False indicates free space
    """
    x = np.arange(0, 200, 0.5)
    y = np.arange(0, 400, 0.5)
    X, Y = np.meshgrid(x, y)

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
        in_half_plane(X, Y,  1, 0, 155) &
        in_half_plane(X, Y,  0,-1, -290) &
        in_half_plane(X, Y,  0, 1, 295)
    )

    theta = np.deg2rad(30)
    x0 = 0
    y0 = 38.4
    L1 = 142.25
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
        r_cells = int(np.floor(r / 0.5))
        expanded = np.zeros_like(mask, dtype=bool)

        rows, cols = mask.shape

        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di**2 + dj**2 > r_cells**2:
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

    return all_collisions

# -----------------------------------------------------------
def generate_possible_moves(c2c, actual_state, goal_pos, all_collisions,
                            RPM1, RPM2, delta_time, wheel_radius, wheel_distance):
    """
    Evaluates the current node and generates all valid next states using
    differential drive action primitives, appending them to the open list.

    Inputs:
        c2c             : current cost-to-come for the node
        actual_state    : current state (x, y, theta_deg) in cm and degrees
        goal_pos        : goal position (x, y, theta_deg) in cm and degrees
        all_collisions  : boolean collision map
        RPM1, RPM2      : discrete wheel speeds (RPM)
        delta_time      : duration of each action (seconds)
        wheel_radius    : wheel radius in cm
        wheel_distance  : distance between wheels in cm

    Returns:
        None (updates open_list and open_cost in-place)
    """
    global open_list

    actions_set = run_actions(
        actual_state[0], actual_state[1], actual_state[2],
        RPM1, RPM2, all_collisions, delta_time, wheel_radius, wheel_distance
    )

    for action in actions_set:
        if action is not None:
            x, y, theta, D = action
            new_state = (x, y, theta)

            new_state_disc = discretize(new_state)
            new_c2c = c2c + D
            ct = new_c2c + np.sqrt((x - goal_pos[0])**2 + (y - goal_pos[1])**2)

            new_state_disc_c = grid_index(new_state)
            x_c, y_c, _ = new_state_disc_c

            if x_c < 0 or x_c >= all_collisions.shape[1] or y_c < 0 or y_c >= all_collisions.shape[0]:
                continue

            if all_collisions[y_c, x_c]:
                continue

            if new_state_disc not in open_cost or open_cost.get(new_state_disc)[1] > new_c2c:
                open_cost[new_state_disc] = (actual_state, new_c2c, new_state)
                heapq.heappush(open_list, (ct, (new_state_disc, new_state)))

# -----------------------------------------------------------
def recover_pose_path(final_loc):
    """
    Reconstructs the sequence of moves that enable 
    the robot to go from the initial to goal location

    Inputs:
        final_loc : tuple of the form (parent, cost_to_come, child),
                    retrieved from open_cost, representing the final
                    node in the search tree

    Returns:
        absolute pose path: [(x0,y0,theta0_deg), (x1,y1,theta1_deg), ...]
                            ordered from start to goal
    """
    parent, _, child = final_loc
    order = [child]

    while parent is not None:
        order.append(parent)
        parent, _, child = open_cost[discretize(parent)]

    return order[::-1]

# -----------------------------------------------------------
def poses_to_deltas(pose_path):
    """
    Convert absolute pose path to Falcon output:
    [[dx, dy, dtheta_rad], ...]

    Inputs:
        pose_path : list of absolute poses in the form
                    [(x0, y0, theta0_deg), (x1, y1, theta1_deg), ...]
                    where x, y are in cm and theta is in degrees

    Returns:
        deltas : list of relative motions between consecutive poses,
                 formatted as [[dy, dx, dtheta_rad], ...],
                 where dx, dy are in cm and dtheta is in radians
    """
    deltas = []

    for i in range(1, len(pose_path)):
        x0, y0, t0_deg = pose_path[i - 1]
        x1, y1, t1_deg = pose_path[i]

        dx = (x1 - x0) 
        dy = (y1 - y0) 

        dt_deg = (t1_deg - t0_deg + 180.0) % 360.0 - 180.0
        dtheta = math.radians(dt_deg)

        deltas.append([dy, dx, dtheta])

    return deltas

# -----------------------------------------------------------
def plan_path(start,
              end,
              robot_radius,
              clearance,
              delta_time,
              goal_threshold,
              wheel_radius,
              wheel_distance,
              rpm1,
              rpm2,
              logger):
    """
    Falcon-compatible wrapper

    Inputs:
        start, end            : (x, y, theta) in cm, cm, rad
        robot_radius          : cm
        clearance             : cm
        delta_time            : sec
        goal_threshold        : cm
        wheel_radius          : cm
        wheel_distance        : cm
        rpm1, rpm2            : RPM
        logger                : inernal logging in falconsim

    Returns:
        path = [[dx, dy, dtheta], ...] in cm, cm, rad
    """
    global open_list

    reset_search_state()
    
    start_pos = (float(start[0]), float(start[1]), math.degrees(float(start[2])))
    goal_pos = (float(end[0]), float(end[1]), math.degrees(float(end[2])))

    # total clearance = obstacle clearance + robot radius
    clearance_total_cm = float(clearance) + float(robot_radius)

    all_collisions = build_collision_map(clearance_total_cm)

    # collision check for start and goal
    sx_d, sy_d, _ = grid_index(start_pos)
    gx_d, gy_d, _ = grid_index(goal_pos)

    # if sx_d < 0 or sx_d >= all_collisions.shape[1] or sy_d < 0 or sy_d >= all_collisions.shape[0]:
    #     raise ValueError("Start state is outside map bounds.")
    # if gx_d < 0 or gx_d >= all_collisions.shape[1] or gy_d < 0 or gy_d >= all_collisions.shape[0]:
    #     raise ValueError("Goal state is outside map bounds.")
    # if all_collisions[sy_d, sx_d]:
    #     raise ValueError("Start state is in collision.")
    # if all_collisions[gy_d, gx_d]:
    #     raise ValueError("Goal state is in collision.")

    child_t = tuple(start_pos)
    child_t_disc = discretize(child_t)

    c2c = 0.0
    ct = c2c + np.sqrt((start_pos[0] - goal_pos[0])**2 + (start_pos[1] - goal_pos[1])**2)

    open_cost[child_t_disc] = (None, c2c, start_pos)
    heapq.heappush(open_list, (ct, (child_t_disc, child_t)))

    final_loc = None

    while open_list:
        _, (child_t_disc, child_t) = heapq.heappop(open_list)

        if child_t_disc not in open_cost:
            continue

        parent_disc_state, c2c, actual_state = open_cost[child_t_disc]

        if (actual_state[0] - goal_pos[0])**2 + (actual_state[1] - goal_pos[1])**2 <= goal_threshold**2:
            final_loc = open_cost.get(child_t_disc)
            break

        if child_t_disc not in closed_list:
            closed_list.add(child_t_disc)
            explored_nodes.append(child_t)

            generate_possible_moves(
                c2c, actual_state, goal_pos, all_collisions,
                rpm1, rpm2, delta_time, wheel_radius, wheel_distance
            )

    if final_loc is None:
        return []

    pose_path = recover_pose_path(final_loc)
    path = poses_to_deltas(pose_path)
    logger.info(f"Path length: {len(path)}")
    logger.info(f"Path: {path}")
    return path
