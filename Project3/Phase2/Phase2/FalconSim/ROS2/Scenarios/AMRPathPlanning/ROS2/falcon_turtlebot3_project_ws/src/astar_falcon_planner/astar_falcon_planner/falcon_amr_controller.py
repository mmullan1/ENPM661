import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from .submodules.astar_planner import plan_path

class RobotAStarPlannerNode(Node):
    def __init__(self):
        super().__init__('robot_a_star_planner')
        self.start_time = -1.0
        self.start_delay = 30.0  
        self.get_logger().info(f'READY AStar Planner')
        start = self.declare_parameter(
          'start_position', [0.30, 1.50, 0.0]).get_parameter_value().double_array_value
        end = self.declare_parameter(
          'end_position', [5.00, 1.50, 0.0]).get_parameter_value().double_array_value        
        robot_radius = self.declare_parameter(
          'robot_radius', 0.171).get_parameter_value().double_value
        clearance = self.declare_parameter(
          'clearance', 0.01).get_parameter_value().double_value
        delta_time = self.declare_parameter(
          'delta_time', 4.0).get_parameter_value().double_value
        goal_threshold = self.declare_parameter(
          'goal_threshold', 0.2).get_parameter_value().double_value
        wheel_radius = self.declare_parameter(
          'wheel_radius', 0.033).get_parameter_value().double_value
        wheel_distance = self.declare_parameter(
          'wheel_distance', 0.16).get_parameter_value().double_value
        rpms = self.declare_parameter(
          'rpms', [25.0, 100.0]).get_parameter_value().double_array_value

        #Convert Units m to cm
        int_start = [int(element*100) for element in start]
        int_end = [int(element*100) for element in end]
        robot_radius = robot_radius*100
        clearance = clearance*100
        goal_threshold = goal_threshold*100
        wheel_radius = wheel_radius*100
        wheel_distance = wheel_distance*100   

        path = plan_path(tuple(int_start),tuple(int_end),robot_radius,clearance,delta_time,goal_threshold,wheel_radius,wheel_distance,rpms[0],rpms[1],self.get_logger())
        self.get_logger().info(f'Path 1 Planned')
        actual_stop_pos = int_start
        for move in path:
            actual_stop_pos[0] += move[0]
            actual_stop_pos[1] += move[1]
            actual_stop_pos[2] += move[2]
        actual_stop_pos = [ int(x) for x in actual_stop_pos ]
        
        self.state = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1/30, self.on_timer)
        
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)        
        self.planned_path = path
        self.delta_time = delta_time
        self.current_twist = Twist()
        self.current_twist.angular.x = 0.0
        self.current_twist.angular.y = 0.0
        self.current_twist.angular.z = 0.0
        self.current_twist.linear.x = 0.0
        self.current_twist.linear.y = 0.0
        self.current_twist.linear.z = 0.0
        self.start_time = -1.0
        self.wait_start = 0
        self.last_transform_time_dif = 0.0
        self.get_logger().info(f'READY AStar Node')

    def on_timer(self):
        from_frame_rel = 'IMUSensor_BP_C_0'
        to_frame_rel = 'map'
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return        

        sim_time = t.header.stamp.sec + t.header.stamp.nanosec * 10**-9

        # initialize scene/render reference time
        if self.start_time == -1.0:
            self.get_logger().info('Scene loaded, beginning startup pause')
            self.start_time = sim_time
            return

        # wait before robot starts moving
        if sim_time - self.start_time < self.start_delay:
            self.current_twist.linear.x = 0.0
            self.current_twist.linear.y = 0.0
            self.current_twist.angular.z = 0.0
            self.publisher.publish(self.current_twist)
            return

        # shift effective motion start so timing begins AFTER the delay
        motion_time = sim_time - self.start_delay

        if self.state == 0:
            if len(self.planned_path) > 0:
                if (motion_time - self.start_time) % self.delta_time < self.last_transform_time_dif:
                    self.planned_path.pop(0)

            if len(self.planned_path) > 0:
                self.last_transform_time_dif = (motion_time - self.start_time) % self.delta_time
                self.publish_twist_cmd(self.planned_path)
            else:
                self.get_logger().info('First Target Reached, Waiting')
                self.current_twist.linear.x = 0.0
                self.current_twist.linear.y = 0.0
                self.current_twist.angular.z = 0.0
                self.publisher.publish(self.current_twist)
                self.wait_start = sim_time
                self.state = 1

        elif self.state == 1:
            self.current_twist.linear.x = 0.0
            self.current_twist.linear.y = 0.0
            self.current_twist.angular.z = 0.0
            self.publisher.publish(self.current_twist)


    def publish_twist_cmd(self, planned_path):
        self.current_twist.linear.x = float(planned_path[0][0] / self.delta_time)
        self.current_twist.linear.y = float(planned_path[0][1] / self.delta_time)
        self.current_twist.angular.z = float(planned_path[0][2] / self.delta_time)
        self.publisher.publish(self.current_twist)
    

def main():
    rclpy.init()    
    node = RobotAStarPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

        
