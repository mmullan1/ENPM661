#!/usr/bin/env python3

from typing import List, Optional

import time
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint


class PandaSequence(Node):
    def __init__(self) -> None:
        super().__init__("panda_sequence")

        # Arm uses FollowJointTrajectory
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/panda_arm_controller/follow_joint_trajectory",
        )

        # Gripper uses GripperCommand
        self.hand_client = ActionClient(
            self,
            GripperCommand,
            "/panda_hand_controller/gripper_cmd",
        )

        self.arm_joints = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

    def _duration_msg(self, seconds: float) -> Duration:
        return Duration(
            sec=int(seconds),
            nanosec=int((seconds % 1.0) * 1e9),
        )

    def wait_for_servers(self) -> None:
        self.get_logger().info("Waiting for arm controller action server...")
        if not self.arm_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Arm controller action server not available.")
        self.get_logger().info("Arm controller connected.")

        self.get_logger().info("Waiting for hand controller action server...")
        if not self.hand_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("Hand controller action server not available.")
        self.get_logger().info("Hand controller connected.")

    def send_arm(self, positions: List[float], seconds: float = 3.0) -> None:
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.arm_joints

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = self._duration_msg(seconds)
        goal_msg.trajectory.points = [point]

        self.get_logger().info(f"Sending arm goal: {positions}")
        future = self.arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Arm trajectory goal was rejected.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Arm motion complete.")

    def send_gripper(self, width: float, force: float = 20.0) -> None:
        # For Panda gripper, width is total opening width in meters.
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width
        goal_msg.command.max_effort = force

        self.get_logger().info(f"Sending gripper goal: width={width:.3f}")
        future = self.hand_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Gripper goal was rejected.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Gripper motion complete.")

    def run_sequence(self) -> None:
        # Home state
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Goal 1: one side / extended shape
        goal_1 = [1.0, -1.1, 0.8, -1.6, 0.7, 1.9, 0.2]

        # Goal 2: opposite side / different shape
        goal_2 = [-1.0, -0.6, -0.8, -2.0, -0.6, 1.4, -0.4]

        gripper_open = 0.08
        gripper_closed = 0.00

        self.wait_for_servers()

        self.send_gripper(gripper_open)
        time.sleep(2)
        self.send_arm(home, 3.0)

        self.send_arm(goal_1, 3.0)
        self.send_gripper(gripper_closed)
        time.sleep(2)

        self.send_arm(goal_2, 3.0)
        self.send_gripper(gripper_open)
        time.sleep(2)

        self.send_arm(home, 3.0)


def main() -> None:
    rclpy.init()
    node = PandaSequence()
    try:
        node.run_sequence()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()