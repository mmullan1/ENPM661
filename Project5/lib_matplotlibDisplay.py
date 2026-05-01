
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
        - position: 
        - size:

    Returns:
        - faces: 
        - v: 
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
def load_box_obstacles(yaml_file):
    """
    Loads in the obstacles from the yaml file

    Inputs: 
        - yaml_file: the file with the obstacles (must be in the same directory as the terminal)
    
    Returns: 
        - boxes: 

    
    """
    with open(yaml_file, "r") as f:
        scene = yaml.safe_load(f)

    boxes = []

    for obj in scene["objects"]:
        if obj["type"] == "box":
            faces, vertices = cube_faces(obj["position"], obj["size"])

            mins = vertices.min(axis=0)
            maxs = vertices.max(axis=0)

            boxes.append((mins, maxs))
    
    return boxes

# -------------------------------------------------------------------------------------------------
def unpack_box(box):
    if isinstance(box, dict):
        return np.asarray(box["min"], dtype=float), np.asarray(box["max"], dtype=float)

    if isinstance(box, tuple) or isinstance(box, list):
        return np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)

    raise TypeError(f"Unsupported box format: {type(box)}")

# ================== Geometry Collision Helpers ================
def closest_point_on_box(p, box_min, box_max):
    """
    Closest point on an axis-aligned box to point p.
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
    points = np.asarray(p_list, dtype=float)

    for p in points:
        for mins, maxs in boxes:
            inside_x = mins[0] <= p[0] <= maxs[0]
            inside_y = mins[1] <= p[1] <= maxs[1]
            inside_z = mins[2] <= p[2] <= maxs[2]

            if inside_x and inside_y and inside_z:
                return True

    return False

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

    # Make sure p_list is an array of 3D points
    p_list = np.asarray(p_list, dtype=float)

    # Optional: include base if fk_cr3 does not include it
    # p_list should look like:
    # [base, joint1, joint2, ..., end_effector]
    if p_list.shape[1] != 3:
        raise ValueError("p_list must be an Nx3 array of 3D points.")

    links = build_robot_boundary(p_list, r=link_radius)

    for link in links:
        A = link["start"]
        B = link["end"]
        r = link["radius"]

        for box in boxes:
            box_min, box_max = unpack_box(box)

            if segment_box_collision(A, B, r, box_min, box_max):
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
def expand_rrt(q_start, q_target, max_step_size, steps, boxes):
    """
    Runs the RRT algorithm repeadedly until either a solution is found
    or the steps parameter is exceeded

    Inputs:
        - q_start:
        - q_target:
        - max_step_size: 
        - steps: 
        - boxes:

    Returns:
        - tree: Path from start config to goal config

    """
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
            print("Rejected collision node:", q_candidate)
            continue

        if joint_link_boundaries(q_candidate, boxes, 0.035):
            print("Rejected collision node:", q_candidate)
            continue


        # 5) accept node into tree
        tree.append(q_candidate.copy())
        parents[tuple(q_candidate)] = tuple(q_near)

        if np.linalg.norm(q_candidate - q_target) < max_step_size:
            print("Goal reached")
            return backtrack_path(parents, q_candidate)
        
    return tree

# -------------------------------------------------------------------------------------------------
def backtrack_path(parents, q_end):
    """
    Backtracks from the goal to the start, extracting the
    path that will take me there

    Inputs:
        - parents:
        - q_end:

    Returns: 
        - path: the path from the start to goal config
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
            print("Rejected collision node:", q_interp)
            return True

    return False

# def smooth_tree(tree, boxes):
#     for i in range(len(tree)):
#         for j in range(len(tree)):
#             bool = smooth_node(tree[i], tree[j+1], boxes)
#             if bool is True:
#                 break
#             else:
#                 continue
#         # print(tree[i])

#     return tree

# def smooth_node(tree_s, tree_int, boxes):
#     """
#     Densify a path by interpolating between nodes.

#     Inputs:
#         - tree_s: list of joint configs (path)
#         - tree_int: number of interpolation points between each pair

#     Returns:
#         - new_tree: densified path
#     """
#     new_tree = []

#     for i in range(len(tree_s) - 1):
#         q0 = np.array(tree_s[i], dtype=float)
#         q1 = np.array(tree_s[i + 1], dtype=float)

#         # include start of segment
#         new_tree.append(q0)

#         # interpolate between q0 and q1
#         for s in np.linspace(0, 1, tree_int + 2)[1:-1]:
#             q_interp = q0 + s * (q1 - q0)
#             new_tree.append(q_interp)

#     # include final node
#     new_tree.append(np.array(tree_s[-1], dtype=float))

#     bool = []
#     for i in range(len(new_tree)): 
#         if joint_link_boundaries(new_tree[i], boxes, 0.035):
#             print("Rejected collision node:", new_tree[i])
#             bool.append(True)
            

#     if np.any(bool) is False:
#         bool2 = False
#     else:
#         bool2 = True
#     return bool2
    

# -------------------------------------------------------------------------------------------------
def start_rrt(q_start, q_goal, max_step_size, yaml_file):
    """
    Initializes the RRT algorithm

    Inputs:
        - q_start:
        - q_goal: 
        - max_step_size:
        - yaml_file: 

    returns: 
        - tree: 
    """
    boxes = load_box_obstacles(yaml_file)
    bool = joint_link_boundaries(q_start, boxes)
    if bool is True:
        print("Starting configuration is in collision with environment")
        return

    # expand tree until the goal configuration is reached and return the path
    tree = expand_rrt(q_start, q_goal, max_step_size, 5000, boxes)

    # apply smoothing to the found path
    tree = smooth_tree(tree, boxes)


    return tree


# ============ Visualization Helpers ==========================
def plot_yaml_scene(ax, yaml_file):
    """
    Plots the 3D scene in MatPlotLib

    Inputs:
        - ax:
        - yaml_file:

    Returns: 
        -
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
                alpha=color[3] if len(color) == 4 else 0.35,
                edgecolor="black"
            )

            ax.add_collection3d(cube)

    if all_vertices:
        return np.vstack(all_vertices)

    return np.empty((0, 3))

# -------------------------------------------------------------------------------------------------
def plot_robot_skeleton(ax, q, label, color, alpha=1.0):
    """
    Overlays the initial and final robot "skeleton" 
    in the MatPlotLib environment along with the
    Steps leading to the goal, if found

    Inputs:
        - ax:
        - q: joint configuration of skeleton
        - label: 
        - color: color of the skeletons (different for start, intermediate and goal)
        - alpha: color transparency value

    Returns:
        - p:
    """
    _, p_list, _, T0e = fk_cr3(q)

    p = np.asarray(p_list, dtype=float)
    ee = T0e[:3, 3]

    ax.plot(
        p[:, 0], p[:, 1], p[:, 2],
        marker="o",
        linewidth=3,
        color=color,
        alpha=alpha,
        label=label
    )

    ax.scatter(
        ee[0], ee[1], ee[2],
        s=60,
        color=color,
        alpha=alpha
    )
    return p

# -------------------------------------------------------------------------------------------------
def plot_link_cylinder(ax, p0, p1, radius=0.035, color="tab:green", alpha=0.75, resolution=16):
    """
    Plot a cylinder around a robot link from p0 to p1.
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
    Plot cylindrical collision boundaries around each robot link.
    """

    _, p_list, _, T0e = fk_cr3(q)
    p_list = np.asarray(p_list, dtype=float)

    for i in range(len(p_list) - 1):
        p0 = p_list[i]
        p1 = p_list[i + 1]

        plot_link_cylinder(
            ax,
            p0,
            p1,
            radius=radius,
            color=color,
            alpha=alpha
        )

    return p_list
    
# -------------------------------------------------------------------------------------------------
def set_equal_axes(ax, all_points, pad=0.2):
    """
    Ensures the MatPlotLib scales each axis equally

    Inputs:
        - ax:
        - all_points: 
        - pad: 
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
    Function to overlay everything in MatPlotLib
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
        alpha=0.85
    )

    all_robot_points = [p_base, p_goal]

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

if __name__ == "__main__":
    q_start = [0, 0, 0, 0, 0, 0]
    q_goal = [-33, 58, 69, 38, 87, 64]

    q_new = start_rrt(q_start, q_goal, 30, "lab_scene.yaml")

    render_scene_with_start_goal(
        "lab_scene.yaml",
        q_start,
        q_goal,
        q_new
    )