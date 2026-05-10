import os
import time
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np


class TrajectoryPlayer(Node):
    def __init__(self, csv_file="rrt_path.csv", waypoint_time=5.0):
        super().__init__(f"rrt_trajectory_player_{os.getpid()}")

        self.pub = self.create_publisher(
            JointTrajectory,
            "/cr3_group_controller/joint_trajectory",
            10
        )

        self.csv_file = csv_file
        self.waypoint_time = waypoint_time

        self.path = np.loadtxt(
            self.csv_file,
            delimiter=",",
            skiprows=1
        )

        self.path = np.atleast_2d(self.path)

        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        ]

    def build_trajectory(self):
        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        t = 0.0

        for q in self.path:
            point = JointTrajectoryPoint()

            # degrees -> radians
            point.positions = np.deg2rad(q).tolist()

            t += self.waypoint_time
            point.time_from_start.sec = int(t)
            point.time_from_start.nanosec = int((t % 1.0) * 1e9)

            traj.points.append(point)

        return traj

    def send_trajectory(self):
        traj = self.build_trajectory()

        self.get_logger().info("Waiting for trajectory controller subscriber...")

        start_time = time.time()
        timeout = 5.0

        while self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

            if time.time() - start_time > timeout:
                self.get_logger().warn(
                    "No subscribers found on /cr3_group_controller/joint_trajectory"
                )
                break

        # Publish multiple times so the controller does not miss the message
        for _ in range(5):
            self.pub.publish(traj)
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f"Trajectory sent from {self.csv_file}")


def run_rrt(csv_file="rrt_path.csv", waypoint_time=5.0):
    """
    Loads a saved RRT path CSV and publishes it to the CR3 trajectory controller.
    """

    already_initialized = rclpy.ok()

    if not already_initialized:
        rclpy.init()

    node = TrajectoryPlayer(
        csv_file=csv_file,
        waypoint_time=waypoint_time
    )

    node.send_trajectory()

    # Give ROS time to flush the published messages
    end_time = time.time() + 1.0
    while time.time() < end_time:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()

    if not already_initialized:
        rclpy.shutdown()


def main():
    run_rrt("rrt_path.csv", waypoint_time=5.0)


if __name__ == "__main__":
    main()