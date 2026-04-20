import math
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tkinter import filedialog

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



        filepath = filedialog.asksaveasfilename(
            initialdir="C:/",
            defaultextension=".csv"
        )
        if not filepath:
            return

        self.path = np.atleast_2d(
            np.loadtxt(
                filepath,
                delimiter=',',
                skiprows=1
            )
        )

        print("First 5 CSV waypoints:")
        print(self.path[:5])

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.have_odom = False

        self.goal_tol = 0.05
        self.turn_tol = 0.05
        self.k_lin = 3.0
        self.k_ang = 3.0
        self.max_lin = 1.8
        self.max_ang = 1.8

        self.idx = 0
        self.debug_count = 0
        self.finished = False

        # store actual robot trajectory
        self.actual_path = []

        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        self.x = p.x
        self.y = p.y
        self.yaw = quat_to_yaw(q)
        self.have_odom = True

        # log actual robot position
        self.actual_path.append([self.x, self.y])

    def control_loop(self):
        if not self.have_odom or self.finished:
            return

        cmd = Twist()

        if self.idx >= len(self.path):
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Path complete.')
            self.finished = True
            self.plot_paths()
            return

        gx, gy = self.path[self.idx]
        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)

        self.debug_count += 1
        if self.debug_count % 10 == 0:
            print(
                f"robot: x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f} | "
                f"target[{self.idx}]: gx={gx:.3f}, gy={gy:.3f} | "
                f"dx={dx:.3f}, dy={dy:.3f}, dist={dist:.3f}"
            )

        if dist < self.goal_tol:
            print(f"Reached waypoint {self.idx}: ({gx:.3f}, {gy:.3f})")
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

    def plot_paths(self):
        actual = np.asarray(self.actual_path)

        plt.figure(figsize=(8, 6))

        # desired path
        plt.plot(self.path[:, 0], self.path[:, 1], 'b-', linewidth=2, label='Desired CSV path')
        plt.plot(self.path[:, 0], self.path[:, 1], 'bo', markersize=2)

        # actual robot path
        if len(actual) > 0:
            plt.plot(actual[:, 0], actual[:, 1], 'r-', linewidth=2, label='Actual robot path')

        # mark start and end
        plt.plot(self.path[0, 0], self.path[0, 1], 'go', markersize=8, label='Path start')
        plt.plot(self.path[-1, 0], self.path[-1, 1], 'mo', markersize=8, label='Path end')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('Desired Path vs Actual Robot Path')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    rclpy.init()
    node = PathFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted. Plotting collected trajectory...")
        node.plot_paths()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
