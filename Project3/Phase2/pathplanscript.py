import heapq
import math

def plan_path(start,
              end,
              robot_radius,
              clearance,
              delta_time,
              goal_threshold,
              wheel_radius,
              wheel_distance,
              rpm1,
              rpm2):

    """
    A* planner for a differential-drive robot.

    Inputs
    ------
    start : [x, y, theta]
        Start pose in cm, cm, radians
    end : [x, y, theta]
        Goal pose in cm, cm, radians
    robot_radius : float
        Robot radius in cm
    clearance : float
        Extra clearance in cm
    delta_time : float
        Duration of each action primitive in seconds
    goal_threshold : float
        Goal distance threshold in cm
    wheel_radius : float
        Wheel radius in cm
    wheel_distance : float
        Distance between wheels in cm
    rpm1, rpm2 : float
        Available wheel RPM values

    Returns
    -------
    path : List[List[dx, dy, dtheta]]
        Incremental motion commands in cm and radians
    """

    # --------------------------------------------------
    # Tunable resolution
    XY_RES = 0.5                  # cm
    THETA_RES = math.radians(30)  # rad
    INTEGRATION_DT = 0.1          # sec

    # Map bounds from your earlier code
    MAP_X_MIN, MAP_X_MAX = 0.0, 600.0
    MAP_Y_MIN, MAP_Y_MAX = 0.0, 250.0

    # --------------------------------------------------
    def wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def discretize_state(state):
        x, y, theta = state
        ix = int(round(x / XY_RES))
        iy = int(round(y / XY_RES))
        it = int(round((theta % (2.0 * math.pi)) / THETA_RES)) % int(round((2.0 * math.pi) / THETA_RES))
        return (ix, iy, it)

    def heuristic(state, goal):
        return math.hypot(goal[0] - state[0], goal[1] - state[1])

    def goal_reached(state, goal):
        return math.hypot(goal[0] - state[0], goal[1] - state[1]) <= goal_threshold

    def in_bounds(x, y, margin):
        return ((MAP_X_MIN + margin) <= x <= (MAP_X_MAX - margin) and
                (MAP_Y_MIN + margin) <= y <= (MAP_Y_MAX - margin))

    def is_collision(x, y):
        """
        Hook this into your actual obstacle checker if you have one.
        For now this only checks map boundaries.

        If you already have a global collision function, you can replace this body.
        """
        margin = robot_radius + clearance
        return not in_bounds(x, y, margin)

    def simulate_action(state, ul_rpm, ur_rpm):
        """
        Simulate one differential-drive action for delta_time seconds.

        Returns
        -------
        new_state : (x, y, theta)
        travel_cost : float
            Arc length traveled in cm
        valid : bool
            False if any intermediate point hits collision/out of bounds
        """
        x, y, theta = state

        # RPM -> rad/s
        ul = ul_rpm * 2.0 * math.pi / 60.0
        ur = ur_rpm * 2.0 * math.pi / 60.0

        t = 0.0
        cost = 0.0

        while t < delta_time:
            dt = min(INTEGRATION_DT, delta_time - t)

            v = 0.5 * wheel_radius * (ul + ur)              # cm/s
            w = (wheel_radius / wheel_distance) * (ur - ul) # rad/s

            dx = v * math.cos(theta) * dt
            dy = v * math.sin(theta) * dt
            dtheta = w * dt

            x_new = x + dx
            y_new = y + dy
            theta_new = wrap_to_pi(theta + dtheta)

            if is_collision(x_new, y_new):
                return None, None, False

            cost += math.hypot(dx, dy)

            x, y, theta = x_new, y_new, theta_new
            t += dt

        return (x, y, theta), cost, True

    def reconstruct_pose_path(parent_map, final_key):
        pose_path = []
        k = final_key
        while k is not None:
            pose_path.append(parent_map[k]["state"])
            k = parent_map[k]["parent"]
        pose_path.reverse()
        return pose_path

    def poses_to_deltas(pose_path):
        deltas = []
        for i in range(1, len(pose_path)):
            x0, y0, t0 = pose_path[i - 1]
            x1, y1, t1 = pose_path[i]

            dx = x1 - x0
            dy = y1 - y0
            dtheta = wrap_to_pi(t1 - t0)

            deltas.append([dx, dy, dtheta])
        return deltas

    # --------------------------------------------------
    # Check start/goal validity
    sx, sy, st = float(start[0]), float(start[1]), float(start[2])
    gx, gy, gt = float(end[0]), float(end[1]), float(end[2])

    start_state = (sx, sy, wrap_to_pi(st))
    goal_state = (gx, gy, wrap_to_pi(gt))

    if is_collision(start_state[0], start_state[1]):
        raise ValueError("Start state is in collision or outside valid bounds.")

    if is_collision(goal_state[0], goal_state[1]):
        raise ValueError("Goal state is in collision or outside valid bounds.")

    # Differential-drive action set
    actions = [
        (0, rpm1),
        (rpm1, 0),
        (rpm1, rpm1),
        (0, rpm2),
        (rpm2, 0),
        (rpm2, rpm2),
        (rpm1, rpm2),
        (rpm2, rpm1),
    ]

    # --------------------------------------------------
    # A*
    open_heap = []
    visited_cost = {}
    parent_map = {}

    start_key = discretize_state(start_state)
    visited_cost[start_key] = 0.0
    parent_map[start_key] = {
        "parent": None,
        "state": start_state
    }

    f0 = heuristic(start_state, goal_state)
    heapq.heappush(open_heap, (f0, 0.0, start_key, start_state))

    goal_key_found = None

    while open_heap:
        f_curr, g_curr, curr_key, curr_state = heapq.heappop(open_heap)

        # Skip stale heap entries
        if g_curr > visited_cost.get(curr_key, float("inf")):
            continue

        if goal_reached(curr_state, goal_state):
            goal_key_found = curr_key
            break

        for ul_rpm, ur_rpm in actions:
            next_state, step_cost, valid = simulate_action(curr_state, ul_rpm, ur_rpm)
            if not valid:
                continue

            next_key = discretize_state(next_state)
            new_g = g_curr + step_cost

            if new_g < visited_cost.get(next_key, float("inf")):
                visited_cost[next_key] = new_g
                parent_map[next_key] = {
                    "parent": curr_key,
                    "state": next_state
                }
                new_f = new_g + heuristic(next_state, goal_state)
                heapq.heappush(open_heap, (new_f, new_g, next_key, next_state))

    # --------------------------------------------------
    # No path found
    if goal_key_found is None:
        return []

    # Reconstruct absolute pose path, then convert to [dx, dy, dtheta]
    pose_path = reconstruct_pose_path(parent_map, goal_key_found)
    path = poses_to_deltas(pose_path)

    return path