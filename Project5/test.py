import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def cube_faces(position, size):
    cx, cy, cz = position
    sx, sy, sz = size

    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2

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


def load_scene_from_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        scene = yaml.safe_load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

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

    all_vertices = np.vstack(all_vertices)

    pad = 0.2
    ax.set_xlim(all_vertices[:, 0].min() - pad, all_vertices[:, 0].max() + pad)
    ax.set_ylim(all_vertices[:, 1].min() - pad, all_vertices[:, 1].max() + pad)
    ax.set_zlim(all_vertices[:, 2].min() - pad, all_vertices[:, 2].max() + pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Obstacle Space: {scene.get('frame_id', 'unknown frame')}")

    ax.set_box_aspect([1, 1, 1])

    plt.show()


load_scene_from_yaml("lab_scene.yaml")