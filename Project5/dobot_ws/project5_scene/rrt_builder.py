
# ============== Imports ==================
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib_invK_SDLS import fk_cr3
import argparse
import time
import csv
import io
import contextlib

# import function to move robot in ROS2
from run_rrt import run_rrt

# import function to reset the moveit scene 
from load_scene import reset_moveit_scene

def benchmark_rrt_algorithms(
    q_start,
    q_goal,
    max_step_size,
    scenes,
    algorithms=None,
    runs_per_case=20,
    output_type="2",
    csv_filename="rrt_benchmark_results.csv",
    summary_filename="rrt_benchmark_summary.csv",
    figure_filename="rrt_benchmark_summary.png",
    quiet=True
):
    """
    Runs each RRT algorithm multiple times in multiple environments and tabulates
    the average cost/time along with spread information.

    Inputs:
        - q_start: starting joint configuration
        - q_goal: goal joint configuration
        - max_step_size: maximum joint-space step size
        - scenes: list of YAML scene files to test
        - algorithms: list of algorithms to test
        - runs_per_case: number of times to run each algorithm in each scene
        - output_type: output mode passed into start_rrt()
            "2" is recommended because it returns the smoothed path
        - csv_filename: raw per-run output file
        - summary_filename: summarized statistics output file
        - figure_filename: saved figure table filename
        - quiet: if True, suppresses print output from each individual RRT run

    Returns:
        - summary_rows: list of dictionaries containing summary statistics
    """

    if algorithms is None:
        algorithms = [
            "rrt",
            "rrt_star",
            "birrt",
            "birrt_star",
            "improved_birrt_star"
        ]

    raw_rows = []
    summary_rows = []

    total_runs = len(scenes) * len(algorithms) * runs_per_case

    print("\n========== RRT Benchmark Started ==========")
    print(f"Scenes: {scenes}")
    print(f"Algorithms: {algorithms}")
    print(f"Runs per algorithm per scene: {runs_per_case}")
    print(f"Total runs: {total_runs}")
    print("===========================================\n")

    for scene in scenes:
        for alg in algorithms:

            print(f"\n--- Running {alg} in {scene} ---")

            costs = []
            times = []
            path_lengths = []
            success_flags = []

            for run_idx in range(runs_per_case):

                t0 = time.perf_counter()

                try:
                    if quiet:
                        import os

                        with open(os.devnull, "w") as devnull:
                            with contextlib.redirect_stdout(devnull):
                                path = start_rrt(
                                    q_start,
                                    q_goal,
                                    max_step_size,
                                    scene,
                                    output_type,
                                    alg
                                )
                    else:
                        path = start_rrt(
                            q_start,
                            q_goal,
                            max_step_size,
                            scene,
                            output_type,
                            alg
                        )

                    t1 = time.perf_counter()
                    elapsed_time = t1 - t0

                    if path is None:
                        success = False
                        cost = np.nan
                        path_length = 0

                    else:
                        success = True
                        cost = compute_total_cost(path)
                        path_length = len(path)

                except Exception as e:
                    t1 = time.perf_counter()
                    elapsed_time = t1 - t0

                    success = False
                    cost = np.nan
                    path_length = 0

                    print(f"Run failed: scene={scene}, alg={alg}, run={run_idx + 1}")
                    print(f"Error: {e}")

                costs.append(cost)
                times.append(elapsed_time)
                path_lengths.append(path_length)
                success_flags.append(success)

                raw_rows.append({
                    "scene": scene,
                    "algorithm": alg,
                    "run": run_idx + 1,
                    "success": success,
                    "cost": cost,
                    "time_sec": elapsed_time,
                    "path_nodes": path_length
                })

                if (run_idx + 1) % 5 == 0:
                    print(f"  Completed {run_idx + 1}/{runs_per_case} runs")

            costs = np.asarray(costs, dtype=float)
            times = np.asarray(times, dtype=float)
            path_lengths = np.asarray(path_lengths, dtype=float)
            success_flags = np.asarray(success_flags, dtype=bool)

            successful_costs = costs[success_flags]
            successful_times = times[success_flags]
            successful_path_lengths = path_lengths[success_flags]

            num_success = int(np.sum(success_flags))
            num_fail = runs_per_case - num_success
            success_rate = num_success / runs_per_case

            def safe_stats(data):
                """
                Computes spread statistics safely.
                Returns NaN values if no successful data exists.
                """

                data = np.asarray(data, dtype=float)

                if len(data) == 0:
                    return {
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "q1": np.nan,
                        "median": np.nan,
                        "q3": np.nan,
                        "max": np.nan
                    }

                return {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
                    "min": float(np.min(data)),
                    "q1": float(np.percentile(data, 25)),
                    "median": float(np.median(data)),
                    "q3": float(np.percentile(data, 75)),
                    "max": float(np.max(data))
                }

            cost_stats = safe_stats(successful_costs)
            time_stats = safe_stats(successful_times)
            node_stats = safe_stats(successful_path_lengths)

            summary_row = {
                "scene": scene,
                "algorithm": alg,
                "runs": runs_per_case,
                "successes": num_success,
                "failures": num_fail,
                "success_rate": success_rate,

                "cost_mean": cost_stats["mean"],
                "cost_std": cost_stats["std"],
                "cost_min": cost_stats["min"],
                "cost_q1": cost_stats["q1"],
                "cost_median": cost_stats["median"],
                "cost_q3": cost_stats["q3"],
                "cost_max": cost_stats["max"],

                "time_mean_sec": time_stats["mean"],
                "time_std_sec": time_stats["std"],
                "time_min_sec": time_stats["min"],
                "time_q1_sec": time_stats["q1"],
                "time_median_sec": time_stats["median"],
                "time_q3_sec": time_stats["q3"],
                "time_max_sec": time_stats["max"],

                "path_nodes_mean": node_stats["mean"],
                "path_nodes_std": node_stats["std"],
                "path_nodes_min": node_stats["min"],
                "path_nodes_q1": node_stats["q1"],
                "path_nodes_median": node_stats["median"],
                "path_nodes_q3": node_stats["q3"],
                "path_nodes_max": node_stats["max"]
            }

            summary_rows.append(summary_row)

            print(f"Finished {alg} in {scene}")
            print(f"  Success rate: {success_rate * 100:.1f}%")
            print(f"  Avg cost:     {cost_stats['mean']:.3f}")
            print(f"  Cost std:     {cost_stats['std']:.3f}")
            print(f"  Avg time:     {time_stats['mean']:.3f} sec")
            print(f"  Time std:     {time_stats['std']:.3f} sec")

    # Save raw per-run results
    raw_fieldnames = [
        "scene",
        "algorithm",
        "run",
        "success",
        "cost",
        "time_sec",
        "path_nodes"
    ]

    with open(csv_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fieldnames)
        writer.writeheader()
        writer.writerows(raw_rows)

    # Save summary table
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []

    with open(summary_filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== RRT Benchmark Complete ==========")
    print(f"Raw results saved to:     {csv_filename}")
    print(f"Summary table saved to:   {summary_filename}")
    print(f"Summary figure saved to:  {figure_filename}")
    print("============================================\n")

    plot_benchmark_summary(
        summary_rows,
        save_filename=figure_filename
    )

    return summary_rows


# -------------------------------------------------------------------------------------------------
def plot_benchmark_summary(summary_rows, save_filename="rrt_benchmark_summary.png"):
    """
    Creates a MatPlotLib figure containing the benchmark summary table.

    Inputs:
        - summary_rows: output from benchmark_rrt_algorithms()
        - save_filename: image filename for saving the table figure

    Returns:
        - None
    """

    if not summary_rows:
        print("No benchmark summary rows to plot.")
        return

    columns = [
        "Scene",
        "Algorithm",
        "Success [%]",
        "Cost Mean",
        "Cost Std",
        "Time Mean [s]",
        "Time Std [s]",
        "Nodes Mean",
        "Nodes Std"
    ]

    table_data = []

    for row in summary_rows:
        table_data.append([
            row["scene"],
            row["algorithm"],
            f"{100.0 * row['success_rate']:.1f}",
            f"{row['cost_mean']:.3f}",
            f"{row['cost_std']:.3f}",
            f"{row['time_mean_sec']:.3f}",
            f"{row['time_std_sec']:.3f}",
            f"{row['path_nodes_mean']:.2f}",
            f"{row['path_nodes_std']:.2f}"
        ])

    fig_height = max(3.0, 0.45 * len(table_data) + 1.5)
    fig_width = 15.0

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for col_idx in range(len(columns)):
        table[(0, col_idx)].set_text_props(weight="bold")

    ax.set_title(
        "RRT Benchmark Summary",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    plt.tight_layout()

    if save_filename is not None:
        plt.savefig(save_filename, dpi=300, bbox_inches="tight")
        print(f"Benchmark summary figure saved to: {save_filename}")

    plt.show()
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
# -------------------------------------------------------------------------------------------------
def segment_box_collision(A, B, radius, box_min, box_max, samples=None, sample_spacing=None):
    """
    Approximate capsule-vs-box collision by sampling points along the link axis.

    Collision occurs if any sampled point on the segment is within radius
    of the box.

    Inputs:
        A, B: segment endpoints
        radius: capsule/cylinder radius
        box_min, box_max: AABB bounds
        samples: optional fixed number of checks along the link
        sample_spacing: optional desired distance between samples.
                        If None, defaults to radius.

    Returns:
        bool
    """

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    length = np.linalg.norm(B - A)

    if length < 1e-9:
        closest_box_point = closest_point_on_box(A, box_min, box_max)
        dist_sq = np.sum((A - closest_box_point) ** 2)
        return dist_sq <= radius ** 2

    # If samples is not manually specified, choose it based on link length.
    # This avoids using 25 samples on every link regardless of whether the link is short or long.
    if samples is None:
        if sample_spacing is None:
            sample_spacing = radius

        samples = int(np.ceil(length / sample_spacing)) + 1

        # Clamp the number of samples so it does not become ridiculous.
        samples = max(3, min(samples, 15))

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

            if segment_box_collision(
                A,
                B,
                link_radius,
                box_min,
                box_max,
                samples=None,
                sample_spacing=link_radius
            ):
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
        -alg_type: which algorithm is run
            "rrt": basic rrt algorithm
            "rrt_star": rrt* algorithm
            "rrt_star_lazy": lazy rrt* algorithm
            "preview": only display the scene setup
            "birrt" bidirectional rrt
            "birrt_star": bi-rrt* algorithm
            "improved_birrt_star": improved bi-rrt* algorithm
    returns: 
        - tree: 
    """
    global costs

    # initialize timer
    t0 = time.perf_counter()

    if alg_type == "preview":
        return

    # check to see if starting config is in a collision
    boxes = load_box_obstacles(yaml_file, margin=0.00)
    bool = joint_link_boundaries(q_start, boxes)
    if bool is True:
        print("Starting configuration is in collision with environment")
        return
    bool = joint_link_boundaries(q_goal, boxes)
    if bool is True:
        print("Goal configuration is in collision with environment")
        return


    # expand tree until the goal configuration is reached and return the path
    if alg_type in {"rrt", "rrt_rigorious"}:
        tree = expand_rrt(q_start, q_goal, max_step_size, 5000, boxes, output_type, alg_type)

    elif alg_type in {"rrt_star", "rrt_star_lazy"}:
        tree = expand_rrt(q_start, q_goal, max_step_size, 2000, boxes, output_type, alg_type)


    elif alg_type in {"birrt"}:
        tree = expand_bidirectional_rrt(
            q_start,
            q_goal,
            max_step_size,
            5000,
            boxes,
            alg_type
        )

    elif alg_type in {"birrt_star", "birrt_star_lazy", "improved_birrt_star", "improved_birrt_star_lazy"}:
        tree = expand_bidirectional_rrt(
            q_start,
            q_goal,
            max_step_size,
            2000,
            boxes,
            alg_type
        )

    if tree is None:
        print("No path found.")
        return None

    print(f"Total nodes explored: {len(tree)}")
    if output_type in {"0", "1"}:
        t1 = time.perf_counter()
        print(f"Time taken: {np.round(t1-t0, 3)}")
        return tree

    # apply smoothing to the found path
    tree = smooth_tree(tree, boxes, samples=10)

    # calculate the total cost of the path
    cost = compute_total_cost(tree)
    print(f"Total cost: {cost}")

    t1 = time.perf_counter()
    print(f"Time taken: {np.round(t1-t0, 3)}")
    return tree


# -------------------------------------------------------------------------------------------------
def expand_rrt(q_start, q_target, max_step_size, steps, boxes, output_type, alg_type):
    """
    Executes regular RRT or RRT*.

    RRT:
        Returns the first valid path.

    RRT*:
        Finds an initial path, then continues optimizing for a fixed
        number of extra iterations after the first solution is found.
    """

    if alg_type == "rrt":
        print("Running RRT Algorithm")
        extend_func = extend_tree
        optimize_until_end = False
        extra_optimize_iters = 0
        lazy = None
        use_boundary_node=False,
        
    elif alg_type == "rrt_rigorious":
        print("Running RRT Algorithm")
        extend_func = extend_tree
        optimize_until_end = False
        extra_optimize_iters = 0
        lazy = None
        use_boundary_node=True,

    elif alg_type == "rrt_star":
        print("Running RRT* Algorithm")
        extend_func = extend_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = False
        use_boundary_node=False

    elif alg_type == "rrt_star_lazy":
        print("Running Lazy RRT* Algorithm")
        extend_func = extend_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = True
        use_boundary_node=False

    else:
        raise ValueError(f"Invalid RRT algorithm type: {alg_type}")

    q_start = np.array(q_start, dtype=float)
    q_target = np.array(q_target, dtype=float)

    # initialize tree root and assign it to not have a parent and a cost of 0
    tree = [q_start.copy()]
    parents = {tuple(q_start): None}
    costs = {tuple(q_start): 0.0}

    # assign the best path as none and the best cost as inf
    best_path = None
    best_cost = np.inf

    first_solution_iter = None

    for k in range(steps):

        # ------------------------------------------------------------
        # If RRT* already found a path and completed the extra
        # optimization window, return the best path.
        # ------------------------------------------------------------
        if (
            optimize_until_end
            and first_solution_iter is not None
            and k >= first_solution_iter + extra_optimize_iters
        ):
            print(
                f"{alg_type} stopped after {extra_optimize_iters} "
                f"extra optimization iterations. Best cost: {np.round(best_cost, 3)}"
            )

            if output_type == "0":
                return tree

            return best_path

        # ------------------------------------------------------------
        # Sample random config, sometimes sample goal
        # ------------------------------------------------------------
        if np.random.rand() < 0.1:
            q_rand = q_target.copy()
        else:
            q_rand = np.random.uniform(
                low=-180,
                high=180,
                size=len(q_start)
            )

        # -------------------------------------------------------------
        # Extend tree
        # ------------------------------------------------------------
        q_new = extend_func(
            tree,
            parents,
            costs,
            q_rand,
            max_step_size,
            boxes,
            lazy = lazy, 
            use_boundary_node = use_boundary_node
        
        )

        if q_new is None:
            continue

        # ------------------------------------------------------------
        # Check whether q_new can connect directly to goal
        # ------------------------------------------------------------
        if not smooth_node(q_new, q_target, boxes, samples=5):

            q_new_key = tuple(q_new)
            q_target_key = tuple(q_target)

            # If q_new already is the goal, do NOT set goal's parent to itself.
            if np.allclose(q_new, q_target):
                path = backtrack_path(parents, q_new)
            else:
                parents[q_target_key] = q_new_key
                path = backtrack_path(parents, q_target)

            # Plain RRT returns immediately
            if not optimize_until_end:
                print("RRT connected")

                if output_type == "0":
                    return tree

                return path

            # RRT* records best path and keeps optimizing briefly
            candidate_cost = compute_total_cost(path)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_path = path

                print(
                    f"New best {alg_type} path found at iteration {k} "
                    f"with cost: {np.round(best_cost, 3)}"
                )

            # Mark the first time a solution was found
            if first_solution_iter is None:
                first_solution_iter = k
                print(
                    f"First {alg_type} solution found at iteration {k}. "
                    f"Optimizing for {extra_optimize_iters} more iterations."
                )

    # ------------------------------------------------------------
    # If loop ends naturally, return best RRT* path if one exists
    # ------------------------------------------------------------
    if optimize_until_end and best_path is not None:
        print(f"{alg_type} complete. Best cost: {np.round(best_cost, 3)}")

        if output_type == "0":
            return tree

        return best_path

    print("Solution Not Found!")

    if output_type == "0":
        return tree

    return None
# -------------------------------------------------------------------------------------------------
def expand_bidirectional_rrt(q_start, q_goal, max_step_size, steps, boxes, alg_type):
    """
    Executes Bi-RRT / Bi-RRT* / Lazy Bi-RRT* / Improved Bi-RRT*.

    Bi-RRT:
        Returns the first valid connection.

    Bi-RRT* variants:
        Find an initial path, then continue optimizing for a fixed number
        of extra iterations after the first solution is found.
    """

    if alg_type == "birrt":
        print("Running Bidirectional RRT Algorithm")
        extend_func = extend_tree
        optimize_until_end = False
        extra_optimize_iters = 0
        lazy = False
        use_boundary_node = False

    elif alg_type == "birrt_star":
        print("Running Bidirectional RRT* Algorithm")
        extend_func = extend_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = False
        use_boundary_node = False

    elif alg_type == "birrt_star_lazy":
        print("Running Bidirectional RRT* Algorithm")
        extend_func = extend_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = True
        use_boundary_node = False

    elif alg_type == "improved_birrt_star":
        print("Running Improved Bidirectional RRT* Algorithm")
        extend_func = extend_improved_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = False
        use_boundary_node = False

    elif alg_type == "improved_birrt_star_lazy":
        print("Running Improved Lazy Bidirectional RRT* Algorithm")
        extend_func = extend_improved_tree_star
        optimize_until_end = True
        extra_optimize_iters = 300
        lazy = True
        use_boundary_node = False

    else:
        raise ValueError(f"Invalid bidirectional algorithm type: {alg_type}")

    q_start = np.array(q_start, dtype=float)
    q_goal = np.array(q_goal, dtype=float)

    tree_start = [q_start.copy()]
    tree_goal = [q_goal.copy()]

    costs_start = {tuple(q_start): 0.0}
    costs_goal = {tuple(q_goal): 0.0}

    parents_start = {tuple(q_start): None}
    parents_goal = {tuple(q_goal): None}

    best_path = None
    best_cost = np.inf

    first_solution_iter = None

    for k in range(steps):

        # ------------------------------------------------------------
        # If a Bi-RRT* variant has already found a path and has finished
        # the extra optimization window, stop early.
        # ------------------------------------------------------------
        if (
            optimize_until_end
            and first_solution_iter is not None
            and k >= first_solution_iter + extra_optimize_iters
        ):
            print(
                f"{alg_type} stopped after {extra_optimize_iters} "
                f"extra optimization iterations. Best cost: {np.round(best_cost, 3)}"
            )
            return best_path

        # ------------------------------------------------------------
        # 1) Sample random config, sometimes sample goal
        # ------------------------------------------------------------
        if np.random.rand() < 0.1:
            q_rand = q_goal.copy()
        else:
            q_rand = np.random.uniform(
                low=-180,
                high=180,
                size=len(q_start)
            )

        # ------------------------------------------------------------
        # 2) Extend active tree
        # ------------------------------------------------------------
        q_new_start = extend_func(
            tree_start,
            parents_start,
            costs_start,
            q_rand,
            max_step_size,
            boxes, 
            lazy=lazy,
            use_boundary_node = use_boundary_node
        )

        if q_new_start is None:
            continue

        # ------------------------------------------------------------
        # 3) Extend opposite tree toward new active-tree node
        # ------------------------------------------------------------
        q_new_goal = extend_func(
            tree_goal,
            parents_goal,
            costs_goal,
            q_new_start,
            max_step_size,
            boxes,
            lazy=lazy,
            use_boundary_node = use_boundary_node
        )

        # ------------------------------------------------------------
        # 4) Check whether trees can connect
        # ------------------------------------------------------------
        if q_new_goal is not None:
            if not smooth_node(q_new_start, q_new_goal, boxes, samples=5):

                # Build path with correct tree order
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

                # Enforce path direction: start -> goal
                if np.linalg.norm(path[0] - q_start) > np.linalg.norm(path[-1] - q_start):
                    path.reverse()

                # ------------------------------------------------------------
                # Plain Bi-RRT returns the first valid connection
                # ------------------------------------------------------------
                if not optimize_until_end:
                    print("Bidirectional RRT connected")
                    return path

                # ------------------------------------------------------------
                # Bi-RRT* variants store the best path
                # ------------------------------------------------------------
                candidate_cost = compute_total_cost(path)

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_path = path

                    print(
                        f"New best {alg_type} path found at iteration {k} "
                        f"with cost: {np.round(best_cost, 3)}"
                    )

                # Mark the first time any valid solution is found
                if first_solution_iter is None:
                    first_solution_iter = k
                    print(
                        f"First {alg_type} solution found at iteration {k}. "
                        f"Optimizing for {extra_optimize_iters} more iterations."
                    )

        # ------------------------------------------------------------
        # 5) Swap active and opposite trees
        # ------------------------------------------------------------
        tree_start, tree_goal = tree_goal, tree_start
        parents_start, parents_goal = parents_goal, parents_start
        costs_start, costs_goal = costs_goal, costs_start

    # ------------------------------------------------------------
    # If loop ends naturally, return best path if one exists
    # ------------------------------------------------------------
    if optimize_until_end and best_path is not None:
        print(f"{alg_type} complete. Best cost: {np.round(best_cost, 3)}")
        return best_path

    print("Solution Not Found!")
    return None

# -------------------------------------------------------------------------------------------------
def extend_tree(
    tree,
    parents,
    costs,
    q_target,
    max_step_size,
    boxes,
    lazy=False,
    use_boundary_node=False,
    samples=5
):
    """
    Attempts to grow a tree toward q_target by one step.

    Modes:
        use_boundary_node=False:
            Regular behavior. If q_candidate is in collision or the edge
            to q_candidate crosses collision, reject the extension.

        use_boundary_node=True:
            Boundary-recovery behavior. If q_candidate is in collision or
            the edge crosses collision, try to add the closest valid node
            just before the collision boundary.

    Inputs:
        - tree: list of existing nodes
        - parents: dictionary mapping child node tuple -> parent node tuple
        - costs: dictionary mapping node tuple -> cumulative cost
        - q_target: target/sample configuration
        - max_step_size: maximum extension step
        - boxes: obstacle environment
        - lazy: included for compatibility with extend_tree_star calls
        - use_boundary_node: toggles collision-boundary recovery behavior
        - samples: interpolation samples for smooth_node()

    Returns:
        - q_to_add if a node is added
        - None if extension fails
    """

    q_target = np.array(q_target, dtype=float)

    # ------------------------------------------------------------
    # 1) Find nearest node to q_target
    # ------------------------------------------------------------
    q_near = nearest_node(tree, q_target)

    direction = q_target - q_near
    distance = np.linalg.norm(direction)

    if distance == 0:
        return None

    # ------------------------------------------------------------
    # 2) Step toward q_target
    # ------------------------------------------------------------
    if distance <= max_step_size:
        q_candidate = q_target.copy()
    else:
        q_candidate = q_near + (direction / distance) * max_step_size

    # ------------------------------------------------------------
    # 3) Decide whether candidate/path is valid
    # ------------------------------------------------------------
    candidate_in_collision = joint_link_boundaries(q_candidate, boxes, 0.035)

    edge_in_collision = False

    if not candidate_in_collision:
        edge_in_collision = smooth_node(
            q_near,
            q_candidate,
            boxes,
            samples=samples
        )

    # ------------------------------------------------------------
    # 4) Regular mode: reject collision cases immediately
    # ------------------------------------------------------------
    if not use_boundary_node:
        if candidate_in_collision or edge_in_collision:
            return None

        q_to_add = q_candidate

    # ------------------------------------------------------------
    # 5) Boundary-node mode: recover nearest safe node before collision
    # ------------------------------------------------------------
    else:
        if candidate_in_collision or edge_in_collision:
            q_boundary = find_pre_collision_node(
                q_safe=q_near,
                q_collision=q_candidate,
                boxes=boxes,
                radius=0.035,
                backoff_fraction=0.05,
                max_refine_iters=20
            )

            if q_boundary is None:
                return None

            q_to_add = q_boundary

        else:
            q_to_add = q_candidate

    # ------------------------------------------------------------
    # 6) Add selected node to tree
    # ------------------------------------------------------------
    q_near_key = tuple(q_near)
    q_to_add_key = tuple(q_to_add)

    # Prevent duplicate insertion
    if q_to_add_key in parents:
        return None

    tree.append(q_to_add.copy())
    parents[q_to_add_key] = q_near_key
    costs[q_to_add_key] = costs[q_near_key] + np.linalg.norm(q_to_add - q_near)

    return q_to_add

# -------------------------------------------------------------------------------------------------
def find_pre_collision_node(q_safe, q_collision, boxes, radius=0.035, 
                            backoff_fraction=0.05, max_refine_iters=20):
    """
    Finds a valid node close to the collision boundary.

    Given:
        q_safe      = known valid node already in the tree
        q_collision = candidate configuration that caused collision

    This function performs a binary search along the segment from q_safe
    to q_collision and returns a configuration that is just before the
    collision zone.

    Inputs:
        - q_safe: known collision-free configuration
        - q_collision: configuration found to be in collision
        - boxes: obstacle environment
        - radius: robot link collision radius
        - backoff_fraction: how far to back away from the estimated boundary
        - max_refine_iters: number of binary-search refinements

    Returns:
        - q_boundary_safe: safe node near collision boundary
        - None: if no useful safe node can be found
    """

    q_safe = np.array(q_safe, dtype=float)
    q_collision = np.array(q_collision, dtype=float)

    direction = q_collision - q_safe
    distance = np.linalg.norm(direction)

    if distance == 0:
        return None

    # alpha = 0 means q_safe
    # alpha = 1 means q_collision
    alpha_low = 0.0
    alpha_high = 1.0

    # Binary search for the transition from safe to collision
    for _ in range(max_refine_iters):
        alpha_mid = 0.5 * (alpha_low + alpha_high)
        q_mid = q_safe + alpha_mid * direction

        if joint_link_boundaries(q_mid, boxes, radius):
            alpha_high = alpha_mid
        else:
            alpha_low = alpha_mid

    # alpha_low is the closest known safe point
    # Back off slightly from the collision boundary
    alpha_boundary = alpha_low * (1.0 - backoff_fraction)

    q_boundary_safe = q_safe + alpha_boundary * direction

    # Final safety check
    if joint_link_boundaries(q_boundary_safe, boxes, radius):
        return None

    # Make sure the path from q_safe to this boundary node is also valid
    if smooth_node(q_safe, q_boundary_safe, boxes, samples=5):
        return None

    # Avoid adding a nearly identical duplicate node
    if np.linalg.norm(q_boundary_safe - q_safe) < 1e-6:
        return None

    return q_boundary_safe
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
def extend_tree_star(
    tree,
    parents,
    costs,
    q_target,
    max_step_size,
    boxes,
    rewire_radius=None,
    lazy=False,
    use_boundary_node=False,
    goal_node=None,
    validate_goal_path=False,
    samples=5
):
    """
    Attempts to grow a tree toward q_target using RRT*.

    Modes:
        lazy=False:
            Regular / safer RRT* behavior.
            - Checks q_near -> q_candidate before adding.
            - Checks candidate parent edges before selecting parent.
            - Checks rewire edges before applying rewires.

        lazy=True:
            Aggressive lazy/speculative RRT* behavior.
            - Does NOT check q_near -> q_candidate before adding.
            - Selects best parent by cost only.
            - Rewires by cost only.
            - Optionally validates/repairs the final goal path afterward.

        use_boundary_node=False:
            If q_candidate or the edge to q_candidate is in collision,
            reject the extension.

        use_boundary_node=True:
            If q_candidate or the edge to q_candidate is in collision,
            try to add the closest safe node before collision instead.

    Returns:
        If lazy=True, validate_goal_path=True, and goal_node is provided:
            q_candidate, path, success

        Otherwise:
            q_candidate or None
    """

    if rewire_radius is None:
        rewire_radius = 2 * max_step_size

    q_target = np.array(q_target, dtype=float)

    # ------------------------------------------------------------
    # Helper: update descendant costs after rewiring
    # ------------------------------------------------------------
    def update_descendant_costs(parent_key):
        for child_key, child_parent_key in list(parents.items()):
            if child_parent_key == parent_key:

                if np.isinf(costs.get(parent_key, np.inf)):
                    costs[child_key] = np.inf
                else:
                    costs[child_key] = costs[parent_key] + np.linalg.norm(
                        np.array(child_key, dtype=float)
                        - np.array(parent_key, dtype=float)
                    )

                update_descendant_costs(child_key)

    # ------------------------------------------------------------
    # Helper: avoid parent cycles
    # ------------------------------------------------------------
    def is_ancestor_key(possible_ancestor_key, node_key):
        """
        Returns True if possible_ancestor_key is already above node_key
        in the parent chain.
        """

        current_key = node_key
        visited = set()

        while parents.get(current_key) is not None:

            if current_key in visited:
                return True

            visited.add(current_key)
            current_key = parents[current_key]

            if current_key == possible_ancestor_key:
                return True

        return False

    # ------------------------------------------------------------
    # Helper: validate and repair lazy final path
    # ------------------------------------------------------------
    def validate_and_repair_path(q_end):
        """
        Validates the current path from root to q_end.

        If an edge is in collision:
            1) Break the invalid edge.
            2) Mark the child and its descendants as disconnected.
            3) Try to reconnect that child to another valid parent.
            4) Repeat until the path is valid or repair fails.
        """

        q_end_key = tuple(q_end)

        while True:

            if q_end_key not in parents:
                return None, False

            if np.isinf(costs.get(q_end_key, np.inf)):
                return None, False

            try:
                path = backtrack_path(parents, q_end)
            except RuntimeError:
                return None, False

            bad_edge_index = None

            for i in range(len(path) - 1):
                if smooth_node(path[i], path[i + 1], boxes, samples=samples):
                    bad_edge_index = i
                    break

            if bad_edge_index is None:
                return path, True

            q_bad_parent = path[bad_edge_index]
            q_bad_child = path[bad_edge_index + 1]

            bad_parent_key = tuple(q_bad_parent)
            bad_child_key = tuple(q_bad_child)

            # Break the invalid edge.
            parents[bad_child_key] = None
            costs[bad_child_key] = np.inf
            update_descendant_costs(bad_child_key)

            # Try to reconnect the bad child to a different valid parent.
            best_parent_key = None
            best_cost = np.inf

            for q_possible_parent in tree:
                q_possible_parent = np.array(q_possible_parent, dtype=float)
                possible_parent_key = tuple(q_possible_parent)

                if possible_parent_key == bad_child_key:
                    continue

                if possible_parent_key == bad_parent_key:
                    continue

                if np.isinf(costs.get(possible_parent_key, np.inf)):
                    continue

                # Avoid cycles.
                if is_ancestor_key(bad_child_key, possible_parent_key):
                    continue

                candidate_cost = costs[possible_parent_key] + np.linalg.norm(
                    q_bad_child - q_possible_parent
                )

                if candidate_cost >= best_cost:
                    continue

                # In lazy mode, this is where repair-time collision checking happens.
                if smooth_node(
                    q_possible_parent,
                    q_bad_child,
                    boxes,
                    samples=samples
                ):
                    continue

                best_parent_key = possible_parent_key
                best_cost = candidate_cost

            if best_parent_key is None:
                return None, False

            parents[bad_child_key] = best_parent_key
            costs[bad_child_key] = best_cost
            update_descendant_costs(bad_child_key)

    # ------------------------------------------------------------
    # Helper: optional boundary-node recovery
    # ------------------------------------------------------------
    def recover_boundary_node(q_safe, q_collision):
        """
        Attempts to find a valid node just before collision.
        """

        q_boundary = find_pre_collision_node(
            q_safe=q_safe,
            q_collision=q_collision,
            boxes=boxes,
            radius=0.035,
            backoff_fraction=0.05,
            max_refine_iters=20
        )

        if q_boundary is None:
            return None

        q_boundary_key = tuple(q_boundary)

        # Avoid duplicate insertion.
        if q_boundary_key in parents:
            return None

        return q_boundary

    # ------------------------------------------------------------
    # 1) Find nearest node and generate candidate
    # ------------------------------------------------------------
    q_near = nearest_node(tree, q_target)

    direction = q_target - q_near
    distance = np.linalg.norm(direction)

    if distance == 0:
        if lazy and validate_goal_path and goal_node is not None:
            path, success = validate_and_repair_path(goal_node)
            return None, path, success

        return None

    if distance <= max_step_size:
        q_candidate = q_target.copy()
    else:
        q_candidate = q_near + (direction / distance) * max_step_size

    q_near_key = tuple(q_near)

    # ------------------------------------------------------------
    # 2) Candidate / edge collision handling
    #
    # Regular mode:
    #     Check candidate and edge immediately.
    #
    # Lazy mode:
    #     Check only candidate pose immediately.
    #     Edge collision is skipped intentionally.
    #
    # Boundary-node mode:
    #     If a checked collision is found, try to replace q_candidate
    #     with a safe node just before collision.
    # ------------------------------------------------------------
    candidate_in_collision = joint_link_boundaries(q_candidate, boxes, 0.035)

    edge_in_collision = False

    if not candidate_in_collision and not lazy:
        edge_in_collision = smooth_node(
            q_near,
            q_candidate,
            boxes,
            samples=samples
        )

    if candidate_in_collision or edge_in_collision:

        if not use_boundary_node:
            if lazy and validate_goal_path and goal_node is not None:
                path, success = validate_and_repair_path(goal_node)
                return None, path, success

            return None

        q_boundary = recover_boundary_node(
            q_safe=q_near,
            q_collision=q_candidate
        )

        if q_boundary is None:
            if lazy and validate_goal_path and goal_node is not None:
                path, success = validate_and_repair_path(goal_node)
                return None, path, success

            return None

        q_candidate = q_boundary

    q_candidate_key = tuple(q_candidate)

    # Prevent duplicate insertion.
    if q_candidate_key in parents:
        return None

    # ------------------------------------------------------------
    # 3) Find nearby nodes
    # ------------------------------------------------------------
    nearby_nodes = near_nodes(tree, q_candidate, rewire_radius)

    # ------------------------------------------------------------
    # 4) Choose parent
    # ------------------------------------------------------------
    best_parent_key = q_near_key
    best_cost = costs[q_near_key] + np.linalg.norm(q_candidate - q_near)

    if lazy:
        # Lazy mode:
        # Choose the best parent by cost only.
        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            candidate_cost = costs[q_neighbor_key] + np.linalg.norm(
                q_candidate - q_neighbor
            )

            if candidate_cost < best_cost:
                best_parent_key = q_neighbor_key
                best_cost = candidate_cost

    else:
        # Regular mode:
        # Rank cheaper parent candidates, then collision-check before accepting.
        parent_candidates = []

        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            candidate_cost = costs[q_neighbor_key] + np.linalg.norm(
                q_candidate - q_neighbor
            )

            if candidate_cost < best_cost:
                parent_candidates.append(
                    (candidate_cost, q_neighbor.copy(), q_neighbor_key)
                )

        parent_candidates.sort(key=lambda item: item[0])

        for candidate_cost, q_neighbor, q_neighbor_key in parent_candidates:
            if not smooth_node(
                q_neighbor,
                q_candidate,
                boxes,
                samples=samples
            ):
                best_parent_key = q_neighbor_key
                best_cost = candidate_cost
                break

    # ------------------------------------------------------------
    # 5) Add candidate to tree
    # ------------------------------------------------------------
    tree.append(q_candidate.copy())
    parents[q_candidate_key] = best_parent_key
    costs[q_candidate_key] = best_cost

    # ------------------------------------------------------------
    # 6) Rewire nearby nodes
    # ------------------------------------------------------------
    if lazy:
        # Lazy mode:
        # Rewire by cost only. No collision checking.
        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            # Do not rewire root.
            if parents.get(q_neighbor_key) is None:
                continue

            # Do not rewire candidate to itself.
            if q_neighbor_key == q_candidate_key:
                continue

            # Avoid cycles:
            # If q_neighbor is already an ancestor of q_candidate,
            # then setting q_neighbor's parent to q_candidate would create a cycle.
            if is_ancestor_key(q_neighbor_key, q_candidate_key):
                continue

            new_cost = costs[q_candidate_key] + np.linalg.norm(
                q_neighbor - q_candidate
            )

            old_cost = costs[q_neighbor_key]

            if new_cost < old_cost:
                parents[q_neighbor_key] = q_candidate_key
                costs[q_neighbor_key] = new_cost
                update_descendant_costs(q_neighbor_key)

    else:
        # Regular mode:
        # Only collision-check rewires that would improve cost.
        rewire_candidates = []

        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            # Do not rewire root.
            if parents.get(q_neighbor_key) is None:
                continue

            # Do not rewire candidate to itself.
            if q_neighbor_key == q_candidate_key:
                continue

            # Avoid cycles.
            if is_ancestor_key(q_neighbor_key, q_candidate_key):
                continue

            new_cost = costs[q_candidate_key] + np.linalg.norm(
                q_neighbor - q_candidate
            )

            old_cost = costs[q_neighbor_key]

            if new_cost < old_cost:
                improvement = old_cost - new_cost
                rewire_candidates.append(
                    (new_cost, improvement, q_neighbor.copy(), q_neighbor_key)
                )

        rewire_candidates.sort(key=lambda item: item[1], reverse=True)

        for new_cost, improvement, q_neighbor, q_neighbor_key in rewire_candidates:
            if smooth_node(q_candidate, q_neighbor, boxes, samples=samples):
                continue

            parents[q_neighbor_key] = q_candidate_key
            costs[q_neighbor_key] = new_cost
            update_descendant_costs(q_neighbor_key)

    # ------------------------------------------------------------
    # 7) Optional lazy final-path validation/backtracking repair
    # ------------------------------------------------------------
    if lazy and validate_goal_path and goal_node is not None:
        path, success = validate_and_repair_path(goal_node)
        return q_candidate, path, success

    return q_candidate
# -------------------------------------------------------------------------------------------------
def extend_improved_tree_star(
    tree,
    parents,
    costs,
    q_target,
    max_step_size,
    boxes,
    rewire_radius=None,
    lazy=False,
    use_boundary_node=False,
    samples=5
):
    """
    Improved RRT*-style tree extension using stored cumulative costs.

    Modes:
        lazy=False:
            Regular improved RRT* behavior.
            - Checks q_near -> q_candidate before adding.
            - Checks parent candidate edges before accepting.
            - Checks rewire edges before applying rewires.

        lazy=True:
            Lazy/speculative improved RRT* behavior.
            - Checks only q_candidate itself immediately.
            - Does NOT check q_near -> q_candidate before adding.
            - Chooses parent by cost only.
            - Rewires by cost only.
            - Final path should be collision-checked later.

        use_boundary_node=False:
            Rejects a colliding candidate.

        use_boundary_node=True:
            If a checked collision is found, attempts to add the closest
            valid node before collision using find_pre_collision_node().
    """

    # ------------------------------------------------------------
    # 0) Setup parameters
    # ------------------------------------------------------------
    q_target = np.array(q_target, dtype=float)

    if rewire_radius is None:
        rewire_radius = 2.0 * max_step_size

    p_base = 0.10
    p_max = 0.25
    alpha = 2.0

    local_bias_radius = 0.5 * max_step_size

    q_low = -180.0
    q_high = 180.0

    # ------------------------------------------------------------
    # Helper: avoid parent cycles
    # ------------------------------------------------------------
    def is_ancestor_key(possible_ancestor_key, node_key):
        current_key = node_key
        visited = set()

        while parents.get(current_key) is not None:
            if current_key in visited:
                return True

            visited.add(current_key)
            current_key = parents[current_key]

            if current_key == possible_ancestor_key:
                return True

        return False

    # ------------------------------------------------------------
    # Helper: update descendant costs after rewiring
    # ------------------------------------------------------------
    def update_descendant_costs(parent_key):
        for child_key, child_parent_key in list(parents.items()):
            if child_parent_key == parent_key:
                costs[child_key] = costs[parent_key] + np.linalg.norm(
                    np.array(child_key, dtype=float)
                    - np.array(parent_key, dtype=float)
                )

                update_descendant_costs(child_key)

    # ------------------------------------------------------------
    # Helper: optional boundary-node recovery
    # ------------------------------------------------------------
    def recover_boundary_node(q_safe, q_collision):
        q_boundary = find_pre_collision_node(
            q_safe=q_safe,
            q_collision=q_collision,
            boxes=boxes,
            radius=0.035,
            backoff_fraction=0.05,
            max_refine_iters=20
        )

        if q_boundary is None:
            return None

        q_boundary_key = tuple(q_boundary)

        if q_boundary_key in parents:
            return None

        return q_boundary

    # ------------------------------------------------------------
    # 1) Find nearest node in current tree
    # ------------------------------------------------------------
    q_near = nearest_node(tree, q_target)
    target_distance = np.linalg.norm(q_target - q_near)

    # ------------------------------------------------------------
    # 2) Apply distance-aware dynamic target bias
    # ------------------------------------------------------------
    dof = len(q_target)

    joint_space_diagonal = np.linalg.norm(
        np.full(dof, q_high - q_low, dtype=float)
    )

    if joint_space_diagonal > 0:
        normalized_distance = target_distance / joint_space_diagonal
    else:
        normalized_distance = 1.0

    bias_probability = p_base + (p_max - p_base) * np.exp(
        -alpha * normalized_distance
    )

    closeness_gate = np.exp(-target_distance / (4.0 * max_step_size))

    local_bias_probability = bias_probability * closeness_gate

    if np.random.rand() < local_bias_probability:
        q_target = q_target + np.random.uniform(
            low=-local_bias_radius,
            high=local_bias_radius,
            size=dof
        )

        q_target = np.clip(q_target, q_low, q_high)

        q_near = nearest_node(tree, q_target)
        target_distance = np.linalg.norm(q_target - q_near)

    # ------------------------------------------------------------
    # 3) Compute adaptive rewiring radius
    # ------------------------------------------------------------
    adaptive_radius = max(
        rewire_radius,
        2.0 * max_step_size * np.sqrt(
            np.log(len(tree) + 1) / (len(tree) + 1)
        )
    )

    closeness_radius_boost = 1.0 + 0.5 * closeness_gate
    adaptive_radius = adaptive_radius * closeness_radius_boost

    # ------------------------------------------------------------
    # 4) Extend tree toward q_target
    # ------------------------------------------------------------
    direction = q_target - q_near
    distance = np.linalg.norm(direction)

    if distance == 0:
        return None

    if distance <= max_step_size:
        q_candidate = q_target.copy()
    else:
        q_candidate = q_near + (direction / distance) * max_step_size

    q_near_key = tuple(q_near)

    # ------------------------------------------------------------
    # 5) Candidate / edge collision handling
    #
    # Regular mode:
    #     Check candidate and q_near -> q_candidate edge immediately.
    #
    # Lazy mode:
    #     Check only candidate pose immediately.
    #     Edge checks are intentionally skipped.
    #
    # Boundary-node mode:
    #     If a checked collision is found, replace q_candidate with
    #     the nearest safe node before collision.
    # ------------------------------------------------------------
    candidate_in_collision = joint_link_boundaries(q_candidate, boxes, 0.035)

    edge_in_collision = False

    if not candidate_in_collision and not lazy:
        edge_in_collision = smooth_node(
            q_near,
            q_candidate,
            boxes,
            samples=samples
        )

    if candidate_in_collision or edge_in_collision:

        if not use_boundary_node:
            return None

        q_boundary = recover_boundary_node(
            q_safe=q_near,
            q_collision=q_candidate
        )

        if q_boundary is None:
            return None

        q_candidate = q_boundary

    q_candidate_key = tuple(q_candidate)

    # Prevent duplicate insertion
    if q_candidate_key in parents:
        return None

    # ------------------------------------------------------------
    # 6) Find nearby nodes using adaptive radius
    # ------------------------------------------------------------
    nearby_nodes = near_nodes(tree, q_candidate, adaptive_radius)

    # ------------------------------------------------------------
    # 7) Choose best parent
    # ------------------------------------------------------------
    best_parent_key = q_near_key
    best_cost = costs[q_near_key] + np.linalg.norm(q_candidate - q_near)

    if lazy:
        # Lazy mode:
        # Choose the lowest-cost parent without collision checking.
        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            candidate_cost = costs[q_neighbor_key] + np.linalg.norm(
                q_candidate - q_neighbor
            )

            if candidate_cost < best_cost:
                best_parent_key = q_neighbor_key
                best_cost = candidate_cost

    else:
        # Regular mode:
        # Rank cheaper parent candidates, then collision-check before accepting.
        parent_candidates = []

        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            candidate_cost = costs[q_neighbor_key] + np.linalg.norm(
                q_candidate - q_neighbor
            )

            if candidate_cost < best_cost:
                parent_candidates.append(
                    (candidate_cost, q_neighbor.copy(), q_neighbor_key)
                )

        parent_candidates.sort(key=lambda item: item[0])

        for candidate_cost, q_neighbor, q_neighbor_key in parent_candidates:
            if not smooth_node(
                q_neighbor,
                q_candidate,
                boxes,
                samples=samples
            ):
                best_parent_key = q_neighbor_key
                best_cost = candidate_cost
                break

    # ------------------------------------------------------------
    # 8) Add q_candidate to tree
    # ------------------------------------------------------------
    tree.append(q_candidate.copy())
    parents[q_candidate_key] = best_parent_key
    costs[q_candidate_key] = best_cost

    q_candidate_cost = costs[q_candidate_key]

    # ------------------------------------------------------------
    # 9) Rewire nearby nodes
    # ------------------------------------------------------------
    if lazy:
        # Lazy mode:
        # Rewire by cost only. No collision checking.
        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            # Do not rewire root
            if parents.get(q_neighbor_key) is None:
                continue

            # Do not rewire candidate to itself
            if q_neighbor_key == q_candidate_key:
                continue

            # Avoid cycles:
            # If q_neighbor is already an ancestor of q_candidate,
            # then setting q_neighbor's parent to q_candidate creates a cycle.
            if is_ancestor_key(q_neighbor_key, q_candidate_key):
                continue

            new_cost = q_candidate_cost + np.linalg.norm(
                q_neighbor - q_candidate
            )

            old_cost = costs[q_neighbor_key]

            if new_cost < old_cost:
                parents[q_neighbor_key] = q_candidate_key
                costs[q_neighbor_key] = new_cost
                update_descendant_costs(q_neighbor_key)

    else:
        # Regular mode:
        # Only collision-check rewires that would improve cost.
        rewire_candidates = []

        for q_neighbor in nearby_nodes:
            q_neighbor = np.array(q_neighbor, dtype=float)
            q_neighbor_key = tuple(q_neighbor)

            # Do not rewire root
            if parents.get(q_neighbor_key) is None:
                continue

            # Do not rewire candidate to itself
            if q_neighbor_key == q_candidate_key:
                continue

            # Avoid cycles
            if is_ancestor_key(q_neighbor_key, q_candidate_key):
                continue

            new_cost = q_candidate_cost + np.linalg.norm(
                q_neighbor - q_candidate
            )

            old_cost = costs[q_neighbor_key]

            if new_cost < old_cost:
                improvement = old_cost - new_cost
                rewire_candidates.append(
                    (new_cost, improvement, q_neighbor.copy(), q_neighbor_key)
                )

        rewire_candidates.sort(key=lambda item: item[1], reverse=True)

        for new_cost, improvement, q_neighbor, q_neighbor_key in rewire_candidates:
            if smooth_node(q_candidate, q_neighbor, boxes, samples=samples):
                continue

            parents[q_neighbor_key] = q_candidate_key
            costs[q_neighbor_key] = new_cost
            update_descendant_costs(q_neighbor_key)

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
def smooth_node(q0, q1, boxes, samples=5):
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
    visited = set()

    while current is not None:

        if current in visited:
            raise RuntimeError(f"Parent cycle detected at node: {current}")

        visited.add(current)

        if current not in parents:
            raise RuntimeError(f"Node missing from parents dictionary: {current}")

        path.append(np.array(current, dtype=float))
        current = parents[current]

    path.reverse()
    return path
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

    p_start_cyl = plot_robot_cylinders(
        ax,
        base_q,
        radius=0.035,
        color="tab:blue",
        alpha=0.20
    )

    p_goal_cyl = plot_robot_cylinders(
        ax,
        goal_q,
        radius=0.035,
        color="tab:orange",
        alpha=0.20
    )


    all_robot_points = [p_base, p_goal, p_goal_cyl]
    if new_q is not None:
        for i in range(len(new_q)):
            if i == 0:
                continue
            if i == len(new_q) - 1:
                continue

            p_new = plot_robot_skeleton(
                ax,
                new_q[i],
                label="Intermediate Poses" if i == 1 else None,
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

# -------------------------------------------------------------------------------------------------
def str_to_bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("true", "t", "yes", "y", "1"):
        return True

    if value in ("false", "f", "no", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Expected true or false.")

# -------------------------------------------------------------------------------------------------
def compute_total_cost(tree):
    total_cost = 0.0
    for i in range(len(tree)-1):
        q_current = np.array(tree[i], dtype=float)
        q_next = np.array(tree[i+1], dtype=float)

        segment_cost = np.linalg.norm(q_next - q_current)
        total_cost += segment_cost

    total_cost = np.round(total_cost, 3)
    return total_cost

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CR3 RRT planning and render the obstacle scene."
    )

    parser.add_argument(
        "--scene",
        default="lab_scene_easy.yaml",
        help="YAML scene file to load, e.g. lab_scene.yaml or lab_scene_easy.yaml"
    )

    parser.add_argument(
        "--output",
        default="2",
        choices=["0", "1", "2"],
        help="Output type: 0 = all explored nodes, 1 = found path, 2 = smoothed path"
    )

    parser.add_argument(
        "--alg",
        default="rrt",
        choices=[
            "preview",
            "rrt",
            "rrt_rigorious",
            "rrt_star",
            "rrt_star_lazy"
            "birrt",
            "birrt_star",
            "birrt_star_lazy"
            "improved_birrt_star"
            "improved_birrt_star_lazy"
        ],
        help="Algorithm type: preview, rrt, rrt_star, birrt, birrt_star, or improved_birrt_star"
    )

    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Maximum joint-space step size"
    )

    parser.add_argument(
        "--save",
        default="rrt_path.csv",
        help="Output CSV filename"
    )

    parser.add_argument(
        "--motion",
        type=str_to_bool,
        default=False,
        help="Whether to immediately execute the planned RRT trajectory: true or false"
    )

    parser.add_argument(
        "--waypoint_time",
        type=float,
        default=5.0,
        help="Seconds between trajectory waypoints"
    )

    parser.add_argument(
        "--benchmark",
        type=str_to_bool,
        default=False,
        help="Run benchmark mode instead of a single RRT run: true or false"
    )

    parser.add_argument(
        "--benchmark_runs",
        type=int,
        default=200,
        help="Number of benchmark runs per algorithm per environment"
    )

    args = parser.parse_args()

    q_start = [0, 0, 0, 0, 0, 0]
    q_goal = [-33, 58, 69, 38, 87, 64]
    # q_goal = [0, 0, 0, 0, 0, 0]

    # ------------------------------------------------------------
    # Benchmark mode
    # ------------------------------------------------------------
    if args.benchmark:
        benchmark_rrt_algorithms(
            q_start=q_start,
            q_goal=q_goal,
            max_step_size=args.step,
            scenes=[
                "lab_scene_easy.yaml",
                "lab_scene.yaml"
            ],
            algorithms=[
                # "rrt",
                # "rrt_rigorious",
                # "rrt_star",
                # "rrt_star_lazy",
                # "birrt",
                # "birrt_star",
                # "birrt_star_lazy",
                # "improved_birrt_star"
                "improved_birrt_star_lazy"
            ],
            runs_per_case=args.benchmark_runs,
            output_type="2",
            csv_filename="rrt_benchmark_results.csv",
            summary_filename="rrt_benchmark_summary.csv",
            figure_filename="rrt_benchmark_summary.png",
            quiet=True
        )

        raise SystemExit

    # ------------------------------------------------------------
    # Single-run mode
    # ------------------------------------------------------------
    q_new = start_rrt(
        q_start,
        q_goal,
        args.step,
        args.scene,
        args.output,
        args.alg
    )

    if q_new is not None and args.alg != "preview":
        save_path_csv(q_new, args.save)

        if args.motion:
            reset_moveit_scene(args.scene)

            run_rrt(
                csv_file=args.save,
                waypoint_time=args.waypoint_time
            )

        else:
            print("Motion disabled. Path saved but not executed.")

    render_scene_with_start_goal(
        args.scene,
        q_start,
        q_goal,
        q_new
    )


