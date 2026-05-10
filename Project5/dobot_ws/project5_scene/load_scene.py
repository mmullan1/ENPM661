#!/usr/bin/env python3

import time
import yaml
import rclpy
from rclpy.node import Node

from moveit_msgs.msg import PlanningScene, CollisionObject, ObjectColor
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from geometry_msgs.msg import Pose, Point

from scipy.spatial.transform import Rotation as R
import trimesh


class SceneLoader(Node):
    def __init__(self, scene_file):
        super().__init__("scene_loader_once")

        self.scene_file = scene_file

        self.pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10
        )

    def wait_for_planning_scene_subscriber(self, timeout=5.0):
        """
        Wait until something is subscribed to /planning_scene.

        This reduces the chance that the one-shot scene update is published
        before MoveIt/RViz is actually listening.
        """

        self.get_logger().info("Waiting for /planning_scene subscriber...")

        start_time = time.time()

        while self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() - start_time > timeout:
                self.get_logger().warn(
                    "No subscribers detected on /planning_scene. Publishing anyway."
                )
                return False

        self.get_logger().info("Subscriber detected on /planning_scene")
        return True

    def publish_scene_reliably(self, planning_scene, label="", repeats=5, dt=0.1):
        """
        Publishes a PlanningScene message multiple times while spinning.

        This makes the scene update much less timing-sensitive.
        """

        for _ in range(repeats):
            self.pub.publish(planning_scene)
            rclpy.spin_once(self, timeout_sec=dt)

        if label:
            self.get_logger().info(label)

    def remove_existing_scene(self):
        """
        Removes known environment objects before loading the new YAML scene.

        The remove list should be the union of all object names that can appear
        in any of your scene YAML files. It is safe to request removal of an
        object that is not currently present.
        """

        remove_scene = PlanningScene()
        remove_scene.is_diff = True

        object_names_to_remove = [
            "box_0",
            "box_1",
            "box_2"
        ]

        for name in object_names_to_remove:
            collision_object = CollisionObject()
            collision_object.id = name
            collision_object.operation = CollisionObject.REMOVE

            remove_scene.world.collision_objects.append(collision_object)

        self.publish_scene_reliably(
            remove_scene,
            label="Removed existing MoveIt environment objects",
            repeats=5,
            dt=0.1
        )

    def load_scene(self):
        """
        Loads the selected YAML scene into MoveIt's planning scene.
        """

        with open(self.scene_file, "r") as f:
            data = yaml.safe_load(f)

        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        frame_id = data["frame_id"]

        for obj in data["objects"]:
            obj_type = obj["type"].lower()

            collision_object = CollisionObject()
            collision_object.header.frame_id = frame_id
            collision_object.id = obj["name"]

            pose = Pose()
            pose.position.x = float(obj["position"][0])
            pose.position.y = float(obj["position"][1])
            pose.position.z = float(obj["position"][2])

            quat = R.from_euler("xyz", obj["rotation_rpy"]).as_quat()
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            # ---------- MESH ----------
            if obj_type == "mesh":
                mesh_path = obj["file"]
                mesh_data = trimesh.load_mesh(mesh_path)

                mesh_msg = Mesh()

                scale = obj.get("scale", [1.0, 1.0, 1.0])

                for vertex in mesh_data.vertices:
                    point = Point()
                    point.x = float(vertex[0] * scale[0])
                    point.y = float(vertex[1] * scale[1])
                    point.z = float(vertex[2] * scale[2])
                    mesh_msg.vertices.append(point)

                for face in mesh_data.faces:
                    tri = MeshTriangle()
                    tri.vertex_indices = [
                        int(face[0]),
                        int(face[1]),
                        int(face[2])
                    ]
                    mesh_msg.triangles.append(tri)

                collision_object.meshes.append(mesh_msg)
                collision_object.mesh_poses.append(pose)

            # ---------- BOX ----------
            elif obj_type == "box":
                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [
                    float(obj["size"][0]),
                    float(obj["size"][1]),
                    float(obj["size"][2])
                ]

                collision_object.primitives.append(primitive)
                collision_object.primitive_poses.append(pose)

            else:
                self.get_logger().warn(f"Skipping unknown object type: {obj_type}")
                continue

            # ---------- COLOR ----------
            if "color" in obj:
                color = ObjectColor()
                color.id = obj["name"]
                color.color.r = float(obj["color"][0])
                color.color.g = float(obj["color"][1])
                color.color.b = float(obj["color"][2])
                color.color.a = float(obj["color"][3])

                planning_scene.object_colors.append(color)

            collision_object.operation = CollisionObject.ADD
            planning_scene.world.collision_objects.append(collision_object)

        self.publish_scene_reliably(
            planning_scene,
            label=f"Loaded MoveIt scene from {self.scene_file}",
            repeats=5,
            dt=0.1
        )


def reset_moveit_scene(scene_file):
    """
    Callable function.

    Removes old environment objects and loads the selected YAML scene.
    Safe to call from another script.
    """

    already_initialized = rclpy.ok()

    if not already_initialized:
        rclpy.init()

    node = SceneLoader(scene_file)

    # Wait until MoveIt/RViz is listening, or timeout and publish anyway.
    node.wait_for_planning_scene_subscriber(timeout=5.0)

    # Remove old objects.
    node.remove_existing_scene()

    # Give MoveIt a little time to process the remove messages.
    end_time = time.time() + 0.75
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Add new objects from the selected YAML file.
    node.load_scene()

    # Give MoveIt time to process the add messages.
    end_time = time.time() + 1.5
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()

    if not already_initialized:
        rclpy.shutdown()


def main():
    import sys

    scene_file = "lab_scene.yaml"

    if len(sys.argv) > 1:
        scene_file = sys.argv[1]

    reset_moveit_scene(scene_file)


if __name__ == "__main__":
    main()