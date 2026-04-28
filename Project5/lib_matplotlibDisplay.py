import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib_invK_SDLS import fk_cr3


def cube_faces(position, size):
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


def build_axis_lines(T, length=0.05):
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    return (
        np.vstack((origin, origin + length * x_axis)),
        np.vstack((origin, origin + length * y_axis)),
        np.vstack((origin, origin + length * z_axis)),
    )


def plot_axis_triad(ax, T, length=0.05, alpha=1.0):
    xL, yL, zL = build_axis_lines(T, length)

    ax.plot(xL[:, 0], xL[:, 1], xL[:, 2], color="r", linewidth=2, alpha=alpha)
    ax.plot(yL[:, 0], yL[:, 1], yL[:, 2], color="g", linewidth=2, alpha=alpha)
    ax.plot(zL[:, 0], zL[:, 1], zL[:, 2], color="b", linewidth=2, alpha=alpha)


def plot_robot_skeleton(ax, q, label, color, alpha=1.0):
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

    plot_axis_triad(ax, T0e, length=0.05, alpha=alpha)

    return p


def set_equal_axes(ax, all_points, pad=0.2):
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


def render_scene_with_start_goal(yaml_file, base_q, goal_q):
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


if __name__ == "__main__":
    base_q = [0, 0, 0, 0, 0, 0]
    goal_q = [-33, 58, 69, 38, 87, 64]

    render_scene_with_start_goal(
        "lab_scene.yaml",
        base_q,
        goal_q
    )