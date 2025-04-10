#!/usr/bin/python3

import numpy as np

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point

from f1tenth_mppi.utils import *
from f1tenth_mppi.dynamics_models import KBM

class MPPI(Node):
    def __init__(self):
        super().__init__('MPPI')

        # Setup loggers
        self.info_log = rclpy.logging.get_logger('info')
        self.warn_log = rclpy.logging.get_logger('warn')
        self.error_log = rclpy.logging.get_logger('error')

        # Initialize Parameters
        self.initialize_parameters()
        
        if self.get_parameter("drive_topic").value == None:
            self.error_log.error("No parameters set, use --ros-args --params-file <FILE>")
            exit()

        # Initialize Dynamics Model
        self.model = KBM(self.get_parameter("wheelbase").value,
                         self.get_parameter("min_throttle").value,
                         self.get_parameter("max_throttle").value,
                         self.get_parameter("max_steer").value,
                         self.get_parameter("dt").value)

        # Load Waypoints
        try:
            self.waypoints = load_waypoints(self.get_parameter("waypoint_path").value)
        except Exception as e:
            self.error_log.error("Issue loading waypoints")
            self.error_log.error(e)
            exit()

        # Create Publishers
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped,
                                                self.get_parameter("drive_topic").value,
                                                10)
        self.occupancy_pub_ = self.create_publisher(OccupancyGrid,
                                                self.get_parameter("occupancy_topic").value,
                                                10)
        self.marker_pub_ = self.create_publisher(MarkerArray,
                                                self.get_parameter("marker_topic").value,
                                                10)

        # Create message_filters Subscribers
        self.scan_sub_ = Subscriber(self, LaserScan, self.get_parameter("scan_topic").value)
        self.pose_sub_ = Subscriber(self, Odometry, self.get_parameter("pose_topic").value)

        # Create time synchronized callback
        self.time_sync = ApproximateTimeSynchronizer([self.scan_sub_, self.pose_sub_], 10, 0.1)
        self.time_sync.registerCallback(self.callback)

        self.info_log.info("MPPI node initialized")

    def initialize_parameters(self):
        '''
        Initialize all parameters as None so parameters can be loaded from yaml
        '''
        self.declare_parameters(
            namespace='',
            parameters=[
                ('drive_topic', None),
                ('occupancy_topic', None),
                ('marker_topic', None),
                ('pose_topic', None),
                ('scan_topic', None),
                ('waypoint_path', None),
                ('wheelbase', None),
                ('min_throttle', None),
                ('max_throttle', None),
                ('max_steer', None),
                ('dt', None),
                ('num_trajectories', None),
                ('steps_trajectories', None),
            ])

    def callback(self, scan_msg: LaserScan, pose_msg: Odometry):
        '''
        Time synchronized callback to handle LaserScan and Odometry messages

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
            pose_msg (Odometry): Odometry message from the particle filter
        '''

        self.info_log.info("Recieved scan_msg and pose_msg")

        # TODO: Update parameters

        # TODO: Create Occupancy Grid

        # TODO: Create Cost Map

        # Create Trajectories
        self.info_log.info("Sampling Trajectories")
        trajectories, actions = self.sample_trajectories(self.get_parameter("num_trajectories").value,
                                                         self.get_parameter("steps_trajectories").value)

        # TODO: Evaluate Trajectories

        # TODO: Publish AckermannDriveStamped Message

        return
    
    def create_occupancy_grid(self, scan_msg: LaserScan) -> np.ndarray:
        '''
        Process the LaserScan data into an occupancy grid

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
        Returns:
            occupancy_grid (ndarray): The processed occupancy grid
        '''

        # TODO: Process + Filter scan_msg into output
        occupancy_grid = None

        # TODO: Pubish OccupancyGrid Message

        return occupancy_grid
    
    def create_cost_map(self, ego_position: np.ndarray, occupancy_grid: np.ndarray) -> np.ndarray:
        '''
        Create cost map based on current environment

        Args:
            ego_position: (ndarray): 
            occupancy_grid (ndarray): The processed occupancy grid
        Returns:
            cost_map (ndarray): The cost map
        '''

        # TODO: Create cost_map
        cost_map = None

        return cost_map

    def sample_trajectories(self, num_trajectories: int, steps_trajectories: int):
        '''
        Sample random actions from distribution and generate trajectories using model

        Args:
            num_trajectories (int): The number of trajectories to sample
            steps_trajectories (int): The number of steps for each trajectory
        Returns:
            trajectories (ndarray): (num_trajectories x steps_trajectories x 3) array of trajectories
            actions (ndarray): (num_trajectories x steps_trajectories - 1) array of actions
        '''

        # TODO: Figure out smarter way of sampling trajectories
        v = self.get_parameter("min_throttle").value + self.get_parameter("max_throttle").value * np.random.rand(num_trajectories, steps_trajectories - 1, 1)
        omega = self.get_parameter("max_steer").value * (2 * np.random.rand(num_trajectories, steps_trajectories - 1, 1) - 1)

        actions = np.concatenate((v, omega), axis=2)

        # Sample trajectories
        trajectories = np.zeros((num_trajectories, steps_trajectories, 3))
        for i in range(steps_trajectories - 1):
            trajectories[:, i + 1] = self.model.predict(trajectories[:, i], actions[:, i])

        # Publish a subset of trajectories
        self.publish_trajectories(trajectories)

        return trajectories, actions

    def publish_trajectories(self, points: np.ndarray, color: np.ndarray = np.array([0.0, 0.0, 1.0])):
        '''
        Publish MarkerArray message of lines

        Args:
            points (ndarray): NxMx2 array of points to publish
            color (ndarray): The color of the points
        '''

        # Generate MarkerArray message
        marker_array = MarkerArray()
        for j in range(points.shape[0]):
            for i in range(points.shape[1] - 1):
                marker = Marker()
                marker.header.frame_id = "ego_racecar/laser"
                marker.id = j * points.shape[1] + i
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 1.0
                marker.scale.x = 0.01

                p1 = Point()
                p1.x = points[j, i, 0]
                p1.y = points[j, i, 1]
                p2 = Point()
                p2.x = points[j, i+1, 0]
                p2.y = points[j, i+1, 1]

                marker.points = [p1, p2]
                marker_array.markers.append(marker)

        # Publish MarkerArray Message
        self.marker_pub_.publish(marker_array)

    def publish_markers(self, points: np.ndarray, color: np.ndarray = np.array([1.0, 0.0, 0.0]), ns: str = ""):
        '''
        Publish MarkerArray message of points

        Args:
            points (ndarray): Nx2 array of points to publish
            color (ndarray): The color of the points
        '''
        
        # Generate MarkerArray message
        marker_array = MarkerArray()
        for i in range(len(points)):
            marker = Marker()
            marker.ns = ns
            marker.header.frame_id = "ego_racecar/laser"
            marker.id = i + 1
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = points[i, 0]
            marker.pose.position.y = points[i, 1]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            marker_array.markers.append(marker)

        # Publish MarkerArray Message
        self.marker_pub_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
