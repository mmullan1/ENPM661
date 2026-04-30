import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib_invK_SDLS import fk_cr3

count = 5
now = 0

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


def render_scene_with_start_goal(yaml_file, base_q, goal_q, new_q):
    """
    Function to overlay everything in MatPlotLib

    Inputs:
        - yaml_file: 
        - base_q: 
        - goal_q: 
        - new_q: 
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


    print(new_q)
    for i in range(len(new_q)):
        p_new = plot_robot_skeleton(
            ax,
            new_q[i],
            label="New Pose" if i == 0 else None,
            color="tab:green",
            alpha=0.85
        )

    ax.plot([0, 0.08], [0, 0], [0, 0], color="r", linewidth=2)
    ax.plot([0, 0], [0, 0.08], [0, 0], color="g", linewidth=2)
    ax.plot([0, 0], [0, 0], [0, 0.08], color="b", linewidth=2)

    all_points = np.vstack((scene_pts, p_base, p_goal))
    set_equal_axes(ax, all_points)

    ax.set_title("Obstacle Space + CR3 Start/Goal Skeletons")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()

    plt.show()


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
    tree = expand_rrt(q_start, q_goal, max_step_size, 5000, boxes)
    return tree

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

        # 4) reject collision nodes
        if check_collision(q_candidate, boxes):
            print("Rejected collision node:", q_candidate)
            continue

        # 5) accept node into tree
        tree.append(q_candidate.copy())
        parents[tuple(q_candidate)] = tuple(q_near)

        if np.linalg.norm(q_candidate - q_target) < max_step_size:
            print("Goal reached")
            return backtrack_path(parents, q_candidate)
        
    return tree

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