import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import time

class PathPlayer(Node):
    def __init__(self):
        super().__init__("rrt_path_player")

        self.pub = self.create_publisher(JointState, "/joint_states", 10)

        self.path = np.loadtxt("rrt_path.csv", delimiter=",", skiprows=1)

        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6"
        ]

        self.i = 0
        self.timer = self.create_timer(0.2, self.publish_next)

    def publish_next(self):
        if self.i >= len(self.path):
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Convert degrees to radians if your RViz robot expects radians
        msg.position = np.deg2rad(self.path[self.i]).tolist()

        self.pub.publish(msg)

        self.i += 1

def main():
    rclpy.init()
    node = PathPlayer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
