#!/usr/bin/python3

import numpy as np
from scipy.ndimage import distance_transform_edt

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, PointStamped
import tf2_ros
from rclpy.duration import Duration
import tf2_geometry_msgs

from f1tenth_mppi.utils import *
from f1tenth_mppi.dynamics_models import KBM
from scipy import ndimage

import time

class MPPI(Node):
    def __init__(self):
        super().__init__('MPPI')

        # Setup loggers
        self.info_log = rclpy.logging.get_logger('info')
        self.warn_log = rclpy.logging.get_logger('warn')
        self.error_log = rclpy.logging.get_logger('error')

        # Initialize Parameters
        self.initialize_parameters()
        self.u_mean = np.array([self.get_parameter("min_throttle").value, 0.0]) # Mean for sampling trajectories
        
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
        self.cost_map_pub_ = self.create_publisher(OccupancyGrid,
                                                self.get_parameter("cost_map_topic").value,
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

        
        # Occupancy Grid init
        self.og = OccupancyGrid() # Create your occupancy grid here
        self.og.header.frame_id = "ego_racecar/laser"
        self.og.info.resolution = 0.03
        self.og.info.width = 100
        self.og.info.height = 100
        self.og.info.origin.position.x = 0.0
        self.og.info.origin.position.y = -(self.og.info.height * self.og.info.resolution) / 2

        # Setup the TF2 buffer and listener to capture transforms.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Set up the cost map
        self.cost_map = OccupancyGrid()
        self.cost_map.header.frame_id = "ego_racecar/base_link"
        self.cost_map.info.resolution = 0.03 # this means that every pixel represents 3 cm
        self.cost_map.info.width = 100 
        self.cost_map.info.height = 100 
        self.cost_map.data = np.zeros((self.cost_map.info.height, self.cost_map.info.width), dtype=np.int8).flatten().tolist()
        self.cost_map.info.origin.position.x = 0.0
        self.cost_map.info.origin.position.y = -1.5 # Set origin to the center of the grid
        self.cost_map.info.origin.position.z = 0.0  

        self.dist_cost_mult = self.get_parameter("dist_cost_mult").value
        self.obstacle_weight = self.get_parameter("obstacle_weight").value
        self.raceline_weight = self.get_parameter("raceline_weight").value

        self.info_log.info("MPPI node initialized")

    def initialize_parameters(self):
        '''
        Initialize all parameters as None so parameters can be loaded from yaml
        '''
        self.declare_parameters(
            namespace='',
            parameters=[
                ('visualize', None),
                ('drive_topic', None),
                ('occupancy_topic', None),
                ('cost_map_topic', None),
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
                ('v_sigma', None),
                ('omega_sigma', None),
                ('dist_cost_mult', None),
                ('obstacle_weight', None),
                ('raceline_weight', None),
            ])
        
    def callback(self, scan_msg: LaserScan, pose_msg: Odometry):
        '''
        Time synchronized callback to handle LaserScan and Odometry messages

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
            pose_msg (Odometry): Odometry message from the particle filter
        '''

        self.info_log.info("Recieved scan_msg and pose_msg")

        # Visualize optimal waypoints
        if self.get_parameter("visualize").value:
            waypoints_markers = visualize_waypoints(self.waypoints, self.get_clock().now().to_msg(), color=(1.0, 0.0, 0.0), scale=0.1)
            self.marker_pub_.publish(waypoints_markers)

        # Get the transformation from map frame to local egocar frame
        try:
            self.tf_buffer.lookup_transform("ego_racecar/base_link", "map", rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception as e:
            print(f"not working here")
            print(e)
            return

        # TODO: Update parameters

        # TODO: Create Occupancy Grid
        self.create_occupancy_grid(scan_msg)
        
        # # TODO: Create Cost Map
        self.info_log.info("Creating cost map")
        ego_position = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        occupancy_grid = self.og
        self.update_cost_map(ego_position, occupancy_grid)

        # Create Trajectories
        self.info_log.info("Sampling Trajectories")
        trajectories, actions = self.sample_trajectories(self.get_parameter("num_trajectories").value,
                                                         self.get_parameter("steps_trajectories").value)

        # TODO: Evaluate Trajectories
        best_traj, best_action = self.evaluate_trajectories(trajectories, actions)
        self.publish_trajectories(np.expand_dims(best_traj, 0), np.array([1.0, 0.0, 0.0]))

        # TODO: Update u_mean
        # self.u_mean = best_action[0]

        # TODO: Publish AckermannDriveStamped Message
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = best_action[0, 1]
        drive_msg.drive.speed = best_action[0, 0]
        self.drive_pub_.publish(drive_msg)

        return
    
    def create_occupancy_grid(self, scan_msg: LaserScan) -> np.ndarray:
        '''
        Process the LaserScan data into an occupancy grid

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
        Returns:
            occupancy_grid (ndarray): The processed occupancy grid
        '''

         # Initialize an empty occupancy grid (2D array)
        grid = np.zeros((self.og.info.width, self.og.info.width), dtype=int)

        # Convert laser scan ranges to grid coordinates
        angles = scan_msg.angle_min + scan_msg.angle_increment * np.arange(len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        x_coords = np.round((ranges * np.sin(angles)) / self.og.info.resolution + self.og.info.width / 2).astype(int)
        y_coords = np.round((ranges * np.cos(angles)) / self.og.info.resolution).astype(int)

        # Filter out any points that fall outside the grid boundaries
        valid_mask = (
            (x_coords > 0) & (x_coords < self.og.info.width) &
            (y_coords > 0) & (y_coords < self.og.info.height)
        )

        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Mark occupied cells in the grid
        grid[x_coords, y_coords] = 100

        # Define a simple structuring element (kernel) for dilation
        dilation_kernel = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=bool)

        # Apply binary dilation to expand obstacles
        grid = ndimage.binary_dilation(grid, structure=dilation_kernel).astype(int) * 100
        grid = ndimage.binary_dilation(grid, structure=dilation_kernel).astype(int) * 100

        # Prepare and publish the updated occupancy grid
        self.og.data = grid.flatten().tolist()
        self.og.header.stamp = self.get_clock().now().to_msg()

        if not self.get_parameter("visualize").value:
            return
        
        self.occupancy_pub_.publish(self.og)
    
    def update_cost_map(self, ego_position: np.ndarray, occupancy_grid: np.ndarray) -> np.ndarray:
        '''
        Update cost map based on current environment.
        We want areas that deviate from the raceline to have higher cost.
        Areas that are closer to obstacles in the occupancy_grid should have higher cost.

        Args:
            ego_position: (ndarray): 
            occupancy_grid (ndarray): The processed occupancy grid
        Returns:
            cost_map (ndarray): The cost map
        '''

        # TODO: Create cost_map, should be same shape as occupancy grid
        # OBSTACLE COST
        occupancy_data = np.array(occupancy_grid.data).reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        obstacle_cost = occupancy_data

        raceline_mask = np.zeros_like(occupancy_data, dtype=int)
        waypoints = self.waypoints[:, :2]  # shape: (N, 2)
        transform = self.tf_buffer.lookup_transform("ego_racecar/base_link", "map", rclpy.time.Time(), timeout=Duration(seconds=0.1))

        # 4. Extract translation and rotation (quaternion) from the transform
        trans_x = transform.transform.translation.x
        trans_y = transform.transform.translation.y
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w 
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])

        # Transform all waypoints from 'map' frame to egocar frame:
        points_transformed = np.dot(waypoints, R.T) + np.array([trans_x, trans_y])

        # Convert the transformed points into grid indices.
        ego_x = (points_transformed[:, 0] / self.cost_map.info.resolution)
        ego_y = (points_transformed[:, 1] / self.cost_map.info.resolution) + self.cost_map.info.height / 2

        x_idx = np.round(ego_x).astype(int)
        y_idx = np.round(ego_y).astype(int)

        mask_shape = raceline_mask.shape
        valid = (x_idx >= 0) & (x_idx < mask_shape[0]) & (y_idx >= 0) & (y_idx < mask_shape[1])
        raceline_mask[y_idx[valid], x_idx[valid]] = 100

        dilation_kernel = np.array([[0, 1, 0],
                                    [1, 1, 1],
                                    [0, 1, 0]], dtype=bool)

        # Apply binary dilation to expand obstacles
        raceline_mask = ndimage.binary_dilation(raceline_mask, structure=dilation_kernel).astype(int) * 100

        # Compute the distance from raceline, for each grid cell
        raceline_cost = self.dist_cost_mult * distance_transform_edt(raceline_mask == 0)

        # Final cost map is a weighted sum of the obstacle cost and raceline cost
        cost_map = self.obstacle_weight * obstacle_cost + self.raceline_weight * raceline_cost
        cost_map = np.rint(cost_map)
        cost_map = np.clip(cost_map, 0, 100).astype(int)
        
        self.cost_map.data = cost_map.flatten().tolist() 

        if not self.get_parameter("visualize").value:
            return
        
        self.cost_map_pub_.publish(self.cost_map)


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

        # Sample control values
        v = self.u_mean[0] + np.random.randn(num_trajectories, steps_trajectories - 1, 1) * self.get_parameter("v_sigma").value
        omega = self.u_mean[1] + np.random.randn(num_trajectories, steps_trajectories - 1, 1) * self.get_parameter("omega_sigma").value

        # Limit control values
        v = np.clip(v, self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
        omega = np.clip(omega, -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)

        actions = np.concatenate((v, omega), axis=2)

        # Sample trajectories
        trajectories = np.zeros((num_trajectories, steps_trajectories, 3))
        for i in range(steps_trajectories - 1):
            trajectories[:, i + 1] = self.model.predict_euler(trajectories[:, i], actions[:, i])

        # Publish a subset of trajectories
        self.publish_trajectories(trajectories)

        return trajectories, actions
    
    def evaluate_trajectories(self, trajectories: np.ndarray, actions: np.ndarray):
        '''
        Evaluate trajectories using the cost map

        Args:
            trajectories (np.ndarray): (num_trajectories x steps_trajectories x 3) Sampled trajectories
            actions (np.ndarray): (num_trajectories x steps_trajectories x 2) Sampled actions
        Returns:
            best_traj (np.ndarray): (steps_trajectories x 3) Best trajectory
            best_action (np.ndarray): (steps_trajectories x 2) Best action
        '''

        num_trajectories, steps_trajectories = trajectories.shape[0], trajectories.shape[1]
        cost_map = np.array(self.cost_map.data).reshape((self.cost_map.info.height, self.cost_map.info.width))

        # convert each cost map into cost map frame
        trajectories_pixels = trajectories / self.cost_map.info.resolution

        trajectories_pixels[:, :, 0] = trajectories_pixels[:, :, 0]
        trajectories_pixels[:, :, 1] = trajectories_pixels[:, :, 1] + self.cost_map.info.height/2

        # TODO: Reject trajectories that fall outside of the cost map
        traj_scores = np.sum(cost_map[trajectories_pixels[:, :, 1].astype(int), trajectories_pixels[:, :, 0].astype(int)], axis=1)

        # return lowest cost trajectory
        min_traj_index = np.argmin(traj_scores)

        # print(f"Best trajectory is {min_traj_index}")

        return trajectories[min_traj_index], actions[min_traj_index]

    def publish_trajectories(self, points: np.ndarray, color: np.ndarray = np.array([0.0, 0.0, 1.0])):
        '''
        Publish MarkerArray message of lines

        Args:
            points (ndarray): NxMx2 array of points to publish
            color (ndarray): The color of the points
        '''

        if not self.get_parameter("visualize").value:
            return

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

        if not self.get_parameter("visualize").value:
            return
        
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
