
# ============== Imports ==================
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib_invK_SDLS import fk_cr3

# ============= Scene/Box Construction ==================
def cube_faces(position, size):
    """
    Construct the cube extracted from the yaml file

    Inputs:
        - position: [cx, cy, cz]
            Center position of the box in 3D space.
        - size: [sx, sy, sz]
            Box dimensions along each axis.

    Returns:
        - faces:
            List of 6 faces, where each face contains 4 corner vertices.
            Used by Poly3DCollection for plotting.
        - v:
            Array of the 8 box vertices.
    """
    cx, cy, cz = position
    sx, sy, sz = size

    y0, y1 = cx - sx / 2, cx + sx / 2
    x0, x1 = cy - sy / 2, cy + sy / 2
    x0 = -x0
    x1 = -x1
    z0, z1 = cz - sz / 2, cz + sz / 2

    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])

    faces = [
        [v[i] for i in [0, 1, 2, 3]],
        [v[i] for i in [4, 5, 6, 7]],
        [v[i] for i in [0, 1, 5, 4]],
        [v[i] for i in [2, 3, 7, 6]],
        [v[i] for i in [1, 2, 6, 5]],
        [v[i] for i in [0, 3, 7, 4]],
    ]

    return faces, v

# -------------------------------------------------------------------------------------------------
def load_box_obstacles(yaml_file, margin=0.00):
    """
    Loads box obstacles and inflates them by margin.
    margin is in meters.

    Inputs:
        -yaml_file: the yaml file that contains the obstacle information
        -margin: extra padding around the obstacle

    Returns:
        - boxes:
            List of axis-aligned bounding boxes (AABBs), where each box is
            represented as a tuple (mins, maxs).

            mins: [xmin, ymin, zmin] → lower corner of the box  
            maxs: [xmax, ymax, zmax] → upper corner of the box  

            These bounds are optionally inflated by the specified margin and
            are used for collision detection.
    """
    with open(yaml_file, "r") as f:
        scene = yaml.safe_load(f)

    boxes = []

    for obj in scene["objects"]:
        if obj["type"] == "box":
            faces, vertices = cube_faces(obj["position"], obj["size"])

            mins = vertices.min(axis=0) - margin
            maxs = vertices.max(axis=0) + margin

            boxes.append((mins, maxs))

    return boxes

# -------------------------------------------------------------------------------------------------
def unpack_box(box):
    """
    Standardizes different box formats into (min, max) numpy arrays.

    Inputs:
        - box:
            Can be either:
            1) dict with keys "min" and "max"
            2) tuple/list of (mins, maxs)

    Returns:
        - (box_min, box_max):
            Each is a numpy array [x, y, z] representing the
            lower and upper corners of the axis-aligned box.
    """
    if isinstance(box, dict):
        return np.asarray(box["min"], dtype=float), np.asarray(box["max"], dtype=float)

    if isinstance(box, tuple) or isinstance(box, list):
        return np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)

    raise TypeError(f"Unsupported box format: {type(box)}")

# ================== Geometry Collision Helpers ================
def closest_point_on_box(p, box_min, box_max):
    """
    Closest point on an axis-aligned box to point p.

    Inputs:
        - p: the location on the robot that is being checked for collision
        - box_min: [xmin, ymin, zmin] → lower corner of the box  
        - box_max: [xmax, ymax, zmax] → upper corner of the box  

    Returns:
        p_clamped: the corresponding closest point to p on the box surface
    """

    # Step 1: clamp to upper bounds
    p_clamped_upper = np.minimum(p, box_max)

    # Step 2: clamp to lower bounds
    p_clamped = np.maximum(p_clamped_upper, box_min)

    return p_clamped

# -------------------------------------------------------------------------------------------------
def segment_box_collision(A, B, radius, box_min, box_max, samples=25):
    """
    Approximate capsule-vs-box collision by sampling points along the link axis.

    Collision occurs if any sampled point on the segment is within radius
    of the box.

    Inputs:
        A, B: segment endpoints
        radius: capsule/cylinder radius
        box_min, box_max: AABB bounds
        samples: number of checks along the link

    Returns:
        bool
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    for s in np.linspace(0.0, 1.0, samples):
        p = A + s * (B - A)

        closest_box_point = closest_point_on_box(p, box_min, box_max)

        dist_sq = np.sum((p - closest_box_point) ** 2)

        if dist_sq <= radius ** 2:
            return True

    return False

# -------------------------------------------------------------------------------------------------
def build_robot_boundary(p_list, r):
    """
    Build capsule-like collision boundaries around each robot link.

    Inputs:
        p_list: Nx3 array of joint/link endpoint positions
        r: radius around each link

    Returns:
        links: list of link boundary dictionaries
    """

    links = []

    for i in range(len(p_list) - 1):
        A = np.asarray(p_list[i], dtype=float)
        B = np.asarray(p_list[i + 1], dtype=float)

        length = np.linalg.norm(B - A)

        # Skip zero-length links
        if length < 1e-9:
            continue

        links.append({
            "start": A,
            "end": B,
            "radius": r,
            "length": length,
            "direction": (B - A) / length
        })

    return links

# ================== Robot Collision Checks ================
def check_collision(q, boxes):
    """
    Checks each intermediate node to see if it is in collision with the environment

    Inputs:
        - q: joint configuration
        - boxes: collision zone

    Returns:
        - bool: True if collision occured, False otherwise
    """
    _, p_list, _, _ = fk_cr3(q)

    # modify the FK/DH point list before checking collision
    points = reshape_dh(p_list)

    points = np.asarray(p_list, dtype=float)

    # run collision check, but not for the base link (it's attatched to the floor)
    for p in points[1:]:
        for mins, maxs in boxes:
            inside_x = mins[0] <= p[0] <= maxs[0]
            inside_y = mins[1] <= p[1] <= maxs[1]
            inside_z = mins[2] <= p[2] <= maxs[2]

            if inside_x and inside_y and inside_z:
                return True

    return False

# -------------------------------------------------------------------------------------------------
def reshape_dh(p_list):
    p = np.asarray(p_list, dtype=float)

    offset = np.array([0.0, -0.116, 0.0])  # shift link 2 to better match the real robot
    protrude = 0.05  # 5 cm

    modified_links = []

    def extend_link(A, B, mode="none"):
        """
        Used to make the skeleton better match the real robot geometry

        Inputs:
            - A: the starting point of the link
            - B: the ending point of the link
            - mode: dictates what type of extension occurs
                "none"  -> no extension
                "both"  -> extend both ends
                "A"     -> extend only A backward
                "B"     -> extend only B forward

        Returns:
            - A_ext: extended A
            - B_ext: extended B
        """

        A = np.asarray(A)
        B = np.asarray(B)

        if mode == "none":
            return (A, B)

        d = B - A
        L = np.linalg.norm(d)

        if L < 1e-9:
            return (A, B)

        d_hat = d / L

        A_ext = A
        B_ext = B

        if mode in ("both", "A"):
            A_ext = A - protrude * d_hat

        if mode in ("both", "B"):
            B_ext = B + protrude * d_hat

        return (A_ext, B_ext)
    

    # --- Link 0: base/original first link ---
    modified_links.append(extend_link(p[0], p[1], mode="B"))

    # --- Connector 1: original link to offset link ---
    modified_links.append(extend_link(p[1], p[1] + offset, mode="B"))

    # --- Link 1: offset second link ---
    modified_links.append(extend_link(p[1] + offset, p[2] + offset, mode="None"))

    # --- Connector 2: offset link back to original chain ---
    modified_links.append(extend_link(p[2] + offset, p[2], mode="both"))

    # --- Link 2 ---
    modified_links.append(extend_link(p[2], p[3], mode="None"))

    # --- Link 3 ---
    modified_links.append(extend_link(p[3], p[4], mode="A"))

    # --- Link 4 ---
    modified_links.append(extend_link(p[4], p[5], mode="A"))

    # --- Link 5 ---
    modified_links.append(extend_link(p[5], p[6], mode="None"))

    return modified_links

# -------------------------------------------------------------------------------------------------
def joint_link_boundaries(q, boxes, link_radius=0.035):
    """
    Check whether a robot joint configuration causes collision
    between any robot link boundary and any obstacle box.

    Inputs:
        q: joint configuration
        boxes: list of obstacle boxes
              each box can be:
              {
                  "min": np.array([xmin, ymin, zmin]),
                  "max": np.array([xmax, ymax, zmax])
              }

        link_radius: collision radius around each link skeleton.
                     Use same units as fk_cr3 output and boxes.

    Returns:
        bool: True if collision occurs, False otherwise
    """
    _, p_list, _, T0e = fk_cr3(q)

    links = reshape_dh(p_list)

    for A, B in links[1:]:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        for box in boxes:
            box_min, box_max = unpack_box(box)

            if segment_box_collision(A, B, link_radius, box_min, box_max):
                return True

    return False

# =========== RRT Planning ================
def nearest_node(tree, q_rand):
    """
    Finds the nearest node in the tree to the 
    random configuration that is being expanded

    Inputs: 
        - tree: RRT search tree
        - q_rand: random joint config

    Returns: 
        - tree[nearest_index]: the closest entry in the search tree to the random joint config
    """

    distances = [np.linalg.norm(node - q_rand) for node in tree]
    nearest_index = np.argmin(distances)
    return tree[nearest_index]

# -------------------------------------------------------------------------------------------------
def expand_rrt(q_start, q_target, max_step_size, steps, boxes, output_type, alg_type):
    """
    Runs the RRT algorithm repeadedly until either a solution is found
    or the steps parameter is exceeded

    Inputs:
        - q_start: initial joint configuration
        - q_target: target joint configuration
        - max_step_size: maximum joint step size between steps
        - steps: maximum iterations to run before returning unsuccessfully
        - boxes: collision zone
        - output_type: dictates which of the explored nodes are plotted:
            "0" -> all of them (everything in the search tree), 
            "1": -> the initial path found (not smoothed)
            "2": -> the smoothed path
            other -> default to "2"
        -alg_type: which algorithm is run
            "rrt": basic rrt algorithm
            "rrt_star": rrt* algorithm
            "preview": only display the scene setup
            "birrt" bidirectional rrt; if this is the case, expand_rrt will not be called; expand_bidirectional_rrt() will be instead

    Returns:
        - tree: Path from start config to goal config

    """
    global costs

    if alg_type == "rrt":
        print("Running RRT Algotithm")


    elif alg_type == "rrt_star":
        print("Running RRT* Algotithm")

    elif alg_type == "preview":
        return


    q_start = np.array(q_start, dtype=float)
    q_target = np.array(q_target, dtype=float)

    tree = [q_start.copy()]
    parents = {tuple(q_start): None}

    for _ in range(steps):

        # 1) sample random config, sometimes sample goal
        if np.random.rand() < 0.1:
            q_rand = q_target.copy()
        else:
            q_rand = np.random.uniform(low=-180, high=180, size=len(q_start))

        # 2) find closest node already in tree
        q_near = nearest_node(tree, q_rand)

        # 3) step from q_near toward q_rand
        direction = q_rand - q_near
        distance = np.linalg.norm(direction)

        if distance == 0:
            continue

        if distance <= max_step_size:
            q_candidate = q_rand.copy()
        else:
            q_candidate = q_near + (direction / distance) * max_step_size

        # 4) reject collision nodes (skeleton and boundary)
        if check_collision(q_candidate, boxes):
            # print("Rejected collision node:", q_candidate)
            continue

        if joint_link_boundaries(q_candidate, boxes, 0.035):
            # print("Rejected collision node:", q_candidate)
            continue


        # 5) accept node into tree
        if alg_type == "rrt":
            # print("Running RRT Algotithm")
            tree.append(q_candidate.copy())
            parents[tuple(q_candidate)] = tuple(q_near)

        # 5) accept node into tree
        elif alg_type == "rrt_star":
            # print("Running RRT* Algotithm")
            nearby = near_nodes(tree, q_candidate, radius=60)

            # initialize the best parent as the current parent and compute its cost
            best_parent = q_near
            best_cost = costs[tuple(q_near)] + np.linalg.norm(q_candidate - q_near)

            # check to see if any of the nearby nodes result in a lower cost
            for q_nearby in nearby:
                if not smooth_node(q_nearby, q_candidate, boxes, samples=25):
                    candidate_cost = costs[tuple(q_nearby)] + np.linalg.norm(q_candidate - q_nearby)

                    if candidate_cost < best_cost:
                        best_parent = q_nearby
                        best_cost = candidate_cost

            # add the lowest cost found from the initial and nearby nodes and update "tree"
            tree.append(q_candidate.copy())
            parents[tuple(q_candidate)] = tuple(best_parent)
            costs[tuple(q_candidate)] = best_cost

            # rewire old nodes
            for q_nearby in nearby:
                old_cost = costs[tuple(q_nearby)]
                new_cost = costs[tuple(q_candidate)] + np.linalg.norm(q_nearby - q_candidate)

                if new_cost < old_cost:
                    if not smooth_node(q_candidate, q_nearby, boxes, samples=25):
                        parents[tuple(q_nearby)] = tuple(q_candidate)
                        costs[tuple(q_nearby)] = new_cost

        else:
            print("Invalid algorithm type")
            return

        # stop once a point is reached close to the goal
        if np.linalg.norm(q_candidate - q_target) < max_step_size:
            print("Goal reached")
            tree.append(q_target)
            if output_type == "0":
                return tree
            elif output_type in {"1", "2"}:
                return backtrack_path(parents, q_candidate)
        
    print("Solution Not Found!")
    return tree

# -------------------------------------------------------------------------------------------------
def backtrack_path(parents, q_end):
    """
    Reconstructs a path by following parent pointers from a given node
    back to the root of its tree.

    For regular RRT / RRT*:
        - q_end is typically the node near the goal
        - parents traces back to the start configuration
        - Result is a start → goal path (after reversing)

    For Bidirectional RRT:
        - This is used to extract partial paths from each tree
        - One path is from start → connection point
        - The other is from goal → connection point (later reversed)
        - These are then combined using connect_paths()

    Inputs:
        - parents: dictionary mapping node → parent node
        - q_end: the node to start backtracking from

    Returns: 
        - path: ordered list of configurations from root → q_end
    """
    path = []
    current = tuple(q_end)

    while current is not None:
        path.append(np.array(current))
        current = parents[current]

    path.reverse()
    return path

# -------------------------------------------------------------------------------------------------
def smooth_tree(tree, boxes, samples=25):
    """
    Try to shortcut the RRT path by connecting non-adjacent nodes directly.
    If the direct path is collision-free, remove the intermediate nodes.

    Inputs:
        - tree: The initial tree from the start to goal configuration
        - boxes: the collision zonee to avoid
        - samples: The amount of points the new path is discretized into to ensure that there's no collision along the new path

    Returns:
        - smoothed: the smoothed tree, having removed unnecessary nodes
    """

    smoothed = [np.array(q, dtype=float) for q in tree]

    i = 0
    # there must be future nodes in order for smoothing to be performed
    while i < len(smoothed) - 2:

        # try the furthest possible shortcut first
        j = len(smoothed) - 1

        while j > i + 1:

            collision = smooth_node(smoothed[i], smoothed[j], boxes, samples)

            if not collision:
                # shortcut is valid, remove everything between i and j
                smoothed = smoothed[:i + 1] + smoothed[j:]
                break

            j -= 1

        i += 1

    return smoothed

# -------------------------------------------------------------------------------------------------
def smooth_node(q0, q1, boxes, samples=25):
    """
    Check whether the straight-line joint-space path between q0 and q1 collides.

    Inputs:
        - q0: The starting the joint configuration
        - q1: The joint configuration that is being checked to see if it can be reached directly by moving linearly from q0
        - boxes: the collision zonee to avoid
        - samples: The amount of points the new path is discretized into to ensure that there's no collision along the new path
    
    Returns:
        True  -> collision found
        False -> path is collision-free
    """

    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)

    for s in np.linspace(0, 1, samples):
        q_interp = q0 + s * (q1 - q0)

        if joint_link_boundaries(q_interp, boxes, 0.035):
            # print("Rejected collision node:", q_interp)
            return True

    return False

# -------------------------------------------------------------------------------------------------
def start_rrt(q_start, q_goal, max_step_size, yaml_file, output_type, alg_type):
    """
    Initializes the RRT algorithm

    Inputs:
        - q_start: initial joint configuration
        - q_goal: goal joint configuration
        - max_step_size: maximum each joint can step per iteration
        - yaml_file: the file containing the collision obstacles
        - output_type: dictates which of the explored nodes are plotted:
            "0" -> all of them (everything in the search tree), 
            "1": -> the initial path found (not smoothed)
            "2": -> the smoothed path
            other -> default to "2"

    returns: 
        - tree: 
    """
    global costs
    if alg_type == "preview":
        return
    # initialize cost tracker
    costs = {tuple(q_start): 0.0}

    boxes = load_box_obstacles(yaml_file, margin=0.00)
    bool = joint_link_boundaries(q_start, boxes)
    if bool is True:
        print("Starting configuration is in collision with environment")
        return



    # expand tree until the goal configuration is reached and return the path
    if alg_type in {"rrt", "rrt_star"}:
        tree = expand_rrt(q_start, q_goal, max_step_size, 5000, boxes, output_type, alg_type)

    elif alg_type == "birrt":
        tree = expand_bidirectional_rrt(
            q_start,
            q_goal,
            max_step_size,
            5000,
            boxes
        )

    if output_type in {"0", "1"}:
        return tree

    # apply smoothing to the found path
    tree = smooth_tree(tree, boxes)

    if output_type == "2":
        return tree
    else:
        return tree

# -------------------------------------------------------------------------------------------------
def near_nodes(tree, q_new, radius):
    """
    Returns nearby nodes to check for more optimal path when running RRT*

    Inputs:
        - tree: list of existing nodes (joint configurations) in the RRT
        - q_new: the newly generated node to compare against
        - radius: distance threshold defining the neighborhood

    Returns:
        - nearby_nodes: list of nodes within the specified radius of q_new.
                        These are used in RRT* for choosing a better parent
                        and for rewiring to improve path optimality.
    """
    return [
        node for node in tree
        if np.linalg.norm(node - q_new) <= radius
    ]

# =========== Bidirectional RRT Planning ================
def extend_tree(tree, parents, q_target, max_step_size, boxes):
    """
    Attempts to grow a tree toward a target configuration by one step.

    This function is used in Bidirectional RRT (BiRRT) to incrementally
    extend either the start tree or the goal tree toward a sampled node
    or toward the other tree.

    Steps:
        1) Find the nearest existing node in the tree to q_target
        2) Move from that node toward q_target (bounded by max_step_size)
        3) Reject the candidate if it is in collision or the path is invalid
        4) If valid, add it to the tree and record its parent

    Inputs:
        - tree: list of existing nodes (joint configurations)
        - parents: dictionary mapping each node to its parent node
        - q_target: configuration the tree is trying to extend toward
        - max_step_size: maximum distance allowed per extension step
        - boxes: environment obstacles for collision checking

    Returns:
        - q_candidate: the new node added to the tree if successful
        - None: if extension fails (collision or invalid step)
    """

    q_near = nearest_node(tree, q_target)

    direction = q_target - q_near
    distance = np.linalg.norm(direction)

    if distance == 0:
        return None

    if distance <= max_step_size:
        q_candidate = q_target.copy()
    else:
        q_candidate = q_near + (direction / distance) * max_step_size

    if joint_link_boundaries(q_candidate, boxes, 0.035):
        return None

    if smooth_node(q_near, q_candidate, boxes, samples=10):
        return None

    tree.append(q_candidate.copy())
    parents[tuple(q_candidate)] = tuple(q_near)

    return q_candidate

# -------------------------------------------------------------------------------------------------
def connect_paths(parents_start, parents_goal, q_connect_start, q_connect_goal):
    """
    Connects the two search trees when running bidirectional RRT.

    This function is called once a node from the start tree and a node
    from the goal tree can be connected without collision. It reconstructs
    the full path by backtracking through both trees and concatenating them.

    Inputs:
        - parents_start: parent dictionary for the tree grown from the start
        - parents_goal: parent dictionary for the tree grown from the goal
        - q_connect_start: connection node in the start tree
        - q_connect_goal: connection node in the goal tree

    Returns: 
        - path: full path from start → goal formed by:
                (start → connection) + (connection → goal)
    """

    path_start = backtrack_path(parents_start, q_connect_start)
    path_goal = backtrack_path(parents_goal, q_connect_goal)

    # path_goal currently goes goal -> connection,
    # so reverse it to go connection -> goal
    path_goal.reverse()

    return path_start + path_goal

# -------------------------------------------------------------------------------------------------
def expand_bidirectional_rrt(q_start, q_goal, max_step_size, steps, boxes):
    """
    Executes the Bidirectional Rapidly-Exploring Random Tree (BiRRT) algorithm
    to find a collision-free path between a start and goal configuration.

    This method grows two trees simultaneously:
        - One rooted at the start configuration
        - One rooted at the goal configuration

    At each iteration:
        1) A random configuration q_rand is sampled (with goal bias)
        2) The start tree is extended toward q_rand
        3) The goal tree is then extended toward the newly added node
        4) If both trees can connect without collision, a full path is constructed

    The trees alternate growth directions by swapping roles each iteration,
    allowing efficient exploration of the configuration space.

    Inputs:
        - q_start: starting joint configuration
        - q_goal: goal joint configuration
        - max_step_size: maximum step size when extending the tree
        - steps: maximum number of iterations allowed
        - boxes: obstacle environment for collision checking

    Returns:
        - path: list of configurations from start → goal if a connection is found
        - None: if no valid path is found within the given number of steps

    Notes:
        - extend_tree() handles local expansion and collision rejection
        - smooth_node() ensures the connection between trees is collision-free
        - connect_paths() reconstructs the full path using parent pointers
        - A final direction check ensures the path is always start → goal,
          regardless of which tree completed the connection
    """

    print("Running Bidirectional RRT Algorithm")
    q_start = np.array(q_start, dtype=float)
    q_goal = np.array(q_goal, dtype=float)

    tree_start = [q_start.copy()]
    tree_goal = [q_goal.copy()]

    parents_start = {tuple(q_start): None}
    parents_goal = {tuple(q_goal): None}

    for k in range(steps):

        if np.random.rand() < 0.1:
            q_rand = q_goal.copy()
        else:
            q_rand = np.random.uniform(low=-180, high=180, size=len(q_start))

        q_new_start = extend_tree(
            tree_start,
            parents_start,
            q_rand,
            max_step_size,
            boxes
        )

        if q_new_start is None:
            continue

        q_new_goal = extend_tree(
            tree_goal,
            parents_goal,
            q_new_start,
            max_step_size,
            boxes
        )

        if q_new_goal is not None:
            if not smooth_node(q_new_start, q_new_goal, boxes, samples=10):
                print("Bidirectional RRT connected")

                if k % 2 == 0:
                    path = connect_paths(
                        parents_start,
                        parents_goal,
                        q_new_start,
                        q_new_goal
                    )
                else:
                    path = connect_paths(
                        parents_goal,
                        parents_start,
                        q_new_goal,
                        q_new_start
                    )

                # force path direction: start -> goal
                if np.linalg.norm(path[0] - q_start) > np.linalg.norm(path[-1] - q_start):
                    path.reverse()

                return path

        tree_start, tree_goal = tree_goal, tree_start
        parents_start, parents_goal = parents_goal, parents_start

    print("Solution Not Found!")
    return None

# ============ Visualization Helpers ==========================
def plot_yaml_scene(ax, yaml_file):
    """
    Plots the 3D scene in MatPlotLib by loading box obstacles from a YAML file.

    Inputs:
        - ax: MatPlotLib 3D axis object used for plotting
        - yaml_file: path to the YAML file containing scene object definitions

    Returns: 
        - all_vertices:
            Nx3 numpy array of all vertices from every box in the scene.
            Used later for setting equal axis scaling in the plot.
            Returns an empty array if no objects are present.
    """

    with open(yaml_file, "r") as f:
        scene = yaml.safe_load(f)

    all_vertices = []

    for obj in scene["objects"]:
        if obj["type"] == "box":
            faces, vertices = cube_faces(obj["position"], obj["size"])
            all_vertices.append(vertices)

            color = obj.get("color", [0.5, 0.5, 0.5, 0.35])

            cube = Poly3DCollection(
                faces,
                facecolor=color[:3],
                alpha=0.1,
                edgecolor="black"
            )

            ax.add_collection3d(cube)

    if all_vertices:
        return np.vstack(all_vertices)

    return np.empty((0, 3))

# -------------------------------------------------------------------------------------------------
def plot_robot_skeleton(ax, q, label, color, alpha=1.0):
    """
    Plots the skeleton of the robot (modified from DH skeleton to better match the real robot)
    into the MatPlotLib environment.

    Inputs:
        - ax: MatPlotLib 3D axis object used for plotting
        - q: joint configuration (array-like of joint angles)
        - label: label for the plotted skeleton (used in legend)
        - color: color of the skeleton links and joints
        - alpha: transparency level for the plot

    Returns: 
        - all_points:
            Nx3 numpy array of all endpoints of the plotted links.
            Used later for setting equal axis scaling and combining
            with other plotted elements.
    """

    _, p_list, _, T0e = fk_cr3(q)

    links = reshape_dh(p_list)

    first = True
    all_points = []

    for A, B in links:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        ax.plot(
            [A[0], B[0]],
            [A[1], B[1]],
            [A[2], B[2]],
            marker="o",
            linewidth=3,
            color=color,
            alpha=alpha,
            label=label if first else None
        )

        first = False
        all_points.extend([A, B])

    ee = T0e[:3, 3]
    ax.scatter(ee[0], ee[1], ee[2], s=60, color=color, alpha=alpha)

    return np.asarray(all_points)
# -------------------------------------------------------------------------------------------------
def plot_link_cylinder(ax, p0, p1, radius=0.035, color="tab:green", alpha=0.35, resolution=16):
    """
    Plots a cylindrical surface representing a robot link between two points.

    This function constructs a cylinder aligned along the vector from p0 to p1,
    using an orthonormal basis perpendicular to the link direction. The cylinder
    approximates the physical volume of the link for visualization and debugging
    of collision boundaries.

    Inputs:
        - ax: MatPlotLib 3D axis object used for plotting
        - p0: starting point of the link (3D coordinate)
        - p1: ending point of the link (3D coordinate)
        - radius: radius of the cylinder (link thickness)
        - color: color of the cylinder surface
        - alpha: transparency level of the cylinder
        - resolution: number of angular samples used to approximate the circular cross-section

    Returns:
        - None (plots directly onto the provided axis)
    """

    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    axis = p1 - p0
    length = np.linalg.norm(axis)

    if length < 1e-9:
        return

    n = axis / length

    # Pick a vector not parallel to n
    if abs(n[0]) < 0.9:
        temp = np.array([1.0, 0.0, 0.0])
    else:
        temp = np.array([0.0, 1.0, 0.0])

    # Build two perpendicular radial directions
    u = np.cross(n, temp)
    u = u / np.linalg.norm(u)

    v = np.cross(n, u)
    v = v / np.linalg.norm(v)

    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, length, 2)

    theta_grid, z_grid = np.meshgrid(theta, z)

    X = (
        p0[0]
        + z_grid * n[0]
        + radius * np.cos(theta_grid) * u[0]
        + radius * np.sin(theta_grid) * v[0]
    )
    Y = (
        p0[1]
        + z_grid * n[1]
        + radius * np.cos(theta_grid) * u[1]
        + radius * np.sin(theta_grid) * v[1]
    )
    Z = (
        p0[2]
        + z_grid * n[2]
        + radius * np.cos(theta_grid) * u[2]
        + radius * np.sin(theta_grid) * v[2]
    )

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

# -------------------------------------------------------------------------------------------------
def plot_robot_cylinders(ax, q, radius=0.035, color="tab:green", alpha=0.25):
    """
    Plots cylindrical collision boundaries around each link of the modified robot skeleton.

    This function uses the reshaped DH link representation (which may include
    offsets, connectors, and extensions) and draws a cylinder around each link
    segment to approximate the physical volume of the robot for visualization
    and collision debugging.

    Inputs:
        - ax: MatPlotLib 3D axis object used for plotting
        - q: joint configuration (array-like of joint angles)
        - radius: radius of each cylindrical link (collision thickness)
        - color: color of the cylinders
        - alpha: transparency level of the cylinders

    Returns:
        - all_points:
            Nx3 numpy array of all endpoints of the cylindrical links.
            Used for setting equal axis scaling and combining with other
            plotted elements.
    """

    _, p_list, _, T0e = fk_cr3(q)

    # This now returns [(A, B), (A, B), ...]
    links = reshape_dh(p_list)

    all_points = []

    for A, B in links:
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        plot_link_cylinder(
            ax,
            A,
            B,
            radius=radius,
            color=color,
            alpha=alpha
        )

        all_points.extend([A, B])

    return np.asarray(all_points)
    
# -------------------------------------------------------------------------------------------------
def set_equal_axes(ax, all_points, pad=0.2):
    """
    Ensures the MatPlotLib scales each axis equally so that geometry is not distorted.

    This function computes a bounding cube that contains all provided points
    and applies the same range to the x, y, and z axes. This prevents stretching
    or squashing of the scene, which is especially important for accurately
    visualizing robot geometry and collision volumes.

    Inputs:
        - ax: MatPlotLib 3D axis object to modify
        - all_points: Nx3 array of points (scene + robot) used to determine bounds
        - pad: additional margin added to the bounding box for visual spacing
    """
    all_points = np.asarray(all_points)

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)

    center = 0.5 * (mins + maxs)
    span = np.max(maxs - mins)

    if span == 0:
        span = 1.0

    half = span / 2 + pad

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    ax.set_box_aspect([1, 1, 1])



# ================ Full Rendering Function ===================
def render_scene_with_start_goal(yaml_file, base_q, goal_q, new_q):
    """
    Renders the full 3D scene including obstacles, robot configurations, and optional path.

    This function overlays:
        - The obstacle environment loaded from a YAML file
        - The robot at the start configuration (base_q)
        - The robot at the goal configuration (goal_q)
        - Intermediate robot configurations (new_q), if provided
        - Cylindrical link approximations for collision visualization

    It also ensures consistent axis scaling and adds reference axes for orientation.

    Inputs:
        - yaml_file: path to the YAML file describing the obstacle environment
        - base_q: starting joint configuration of the robot
        - goal_q: goal joint configuration of the robot
        - new_q: list of intermediate configurations (e.g., RRT path),
                 or None if only start/goal should be displayed

    Returns:
        - None (displays the rendered MatPlotLib 3D figure)
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scene_pts = plot_yaml_scene(ax, yaml_file)

    p_base = plot_robot_skeleton(
        ax,
        base_q,
        label="Start Pose",
        color="tab:blue",
        alpha=1.0
    )

    p_goal = plot_robot_skeleton(
        ax,
        goal_q,
        label="Goal Pose",
        color="tab:orange",
        alpha=1.0
    )

    all_robot_points = [p_base, p_goal]

    if new_q is not None:
        for i in range(len(new_q)):
            p_new = plot_robot_skeleton(
                ax,
                new_q[i],
                label="New Pose" if i == 0 else None,
                color="tab:green",
                alpha=0.85
            )

            p_cyl = plot_robot_cylinders(
                ax,
                new_q[i],
                radius=0.035,
                color="tab:green",
                alpha=0.20
            ) 

            all_robot_points.append(p_new)
            all_robot_points.append(p_cyl)

        ax.plot([0, 0.08], [0, 0], [0, 0], color="r", linewidth=2)
        ax.plot([0, 0], [0, 0.08], [0, 0], color="g", linewidth=2)
        ax.plot([0, 0], [0, 0], [0, 0.08], color="b", linewidth=2)

    all_points = np.vstack([scene_pts] + all_robot_points)
    set_equal_axes(ax, all_points)

    ax.set_title("Obstacle Space + CR3 Start/Goal Skeletons + Collision Cylinders")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    plt.show()

# ================ Saving Function ===================
def save_path_csv(path, filename="rrt_path.csv"):
    """
    Saves a joint-space path to a CSV file.

    This function takes a sequence of joint configurations (typically
    generated by RRT, RRT*, or BiRRT) and writes them to a CSV file,
    where each row corresponds to a single configuration.

    The path is assumed to already be ordered from start → goal.

    Inputs:
        - path: list or array of joint configurations (Nx6 for CR3)
        - filename: name of the output CSV file

    Returns:
        - None (writes file to disk)
    """

    path = np.asarray(path, dtype=float)
    

    header = "joint1,joint2,joint3,joint4,joint5,joint6"

    np.savetxt(
        filename,
        path,
        delimiter=",",
        header=header,
        comments=""
    )

    print(f"Saved path to {filename}")



if __name__ == "__main__":
    q_start = [0, 0, 0, 0, 0, 0]
    q_goal = [-33, 58, 69, 38, 87, 64]

    # run the algorithm 
    # options:
    # "0": Display all explored nodes "1": Display the found path "2": display the filtered pagth
    # "preview" only display the scene, don't solve "rrt": run basic rrt algorithm "rrt_star: run rrt* algorithm "birrt": run bidirectional rrt algorithm
    q_new = start_rrt(q_start, q_goal, 5, "lab_scene.yaml", "2", "rrt")

    render_scene_with_start_goal(
        "lab_scene.yaml",
        q_start,
        q_goal,
        q_new
    )

    save_path_csv(q_new, "rrt_path.csv")