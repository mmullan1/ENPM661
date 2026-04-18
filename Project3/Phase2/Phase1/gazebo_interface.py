import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class PathFollower(Node):
    def __init__(self):
        super().__init__('astar_path_follower')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        self.path = np.loadtxt('astar_path.csv', delimiter=',', skiprows=1)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.have_odom = False

        self.goal_tol = 0.05      # m
        self.turn_tol = 0.20      # rad
        self.k_lin = 0.6
        self.k_ang = 1.8
        self.max_lin = 0.18
        self.max_ang = 1.5

        self.idx = 0
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = p.x
        self.y = p.y
        self.yaw = quat_to_yaw(q)
        self.have_odom = True

    def control_loop(self):
        if not self.have_odom:
            return

        cmd = Twist()

        if self.idx >= len(self.path):
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Path complete.')
            return

        gx, gy = self.path[self.idx]
        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)

        if dist < self.goal_tol:
            self.idx += 1
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err = wrap_to_pi(target_yaw - self.yaw)

        if abs(yaw_err) > self.turn_tol:
            cmd.linear.x = 0.0
            cmd.angular.z = max(-self.max_ang, min(self.max_ang, self.k_ang * yaw_err))
        else:
            cmd.linear.x = min(self.max_lin, self.k_lin * dist)
            cmd.angular.z = max(-self.max_ang, min(self.max_ang, self.k_ang * yaw_err))

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()