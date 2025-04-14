#!/usr/bin/python3

import copy
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

import rclpy
import tf2_ros
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.duration import Duration

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

        # Initialize Occupancy Grid
        self.og = OccupancyGrid() # Create your occupancy grid here
        self.og.header.frame_id = self.get_parameter("vehicle_frame").value
        self.og.info.resolution = self.get_parameter("cost_map_res").value
        self.og.info.width = self.get_parameter("cost_map_width").value
        self.og.info.height = self.get_parameter("cost_map_width").value
        self.og.info.origin.position.x = 0.0
        self.og.info.origin.position.y = -(self.og.info.height * self.og.info.resolution) / 2

        # Initialize Cost Map
        self.cost_map = copy.deepcopy(self.og)

        # Load Waypoints
        try:
            self.waypoints = load_waypoints(self.get_parameter("waypoint_path").value)
        except Exception as e:
            self.error_log.error("Issue loading waypoints")
            self.error_log.error(e)
            exit()

        # Setup the TF2 buffer and listener to capture transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        
        # Visualize waypoints
        if self.get_parameter("visualize").value:
            self.publish_markers(self.waypoints, np.array([1.0, 0.0, 0.0]), "waypoints")

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
                ('visualize', None),
                ('waypoint_path', None),
                ('vehicle_frame', None),
                ('drive_topic', None),
                ('occupancy_topic', None),
                ('cost_map_topic', None),
                ('marker_topic', None),
                ('pose_topic', None),
                ('scan_topic', None),
                ('wheelbase', None),
                ('min_throttle', None),
                ('max_throttle', None),
                ('max_steer', None),
                ('dt', None),
                ('num_trajectories', None),
                ('steps_trajectories', None),
                ('v_sigma', None),
                ('omega_sigma', None),
                ('cost_map_width', None),
                ('cost_map_res', None),
                ('dist_cost_mult', None),
                ('obstacle_weight', None),
                ('raceline_weight', None),
                ('obstacle_dilation', None),
                ('raceline_dilation', None),
                ('heading_weight', None),
            ])
        
    def callback(self, scan_msg: LaserScan, pose_msg: Odometry):
        '''
        Time synchronized callback to handle LaserScan and Odometry messages

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
            pose_msg (Odometry): Odometry message from the particle filter
        '''

        self.info_log.info("Recieved scan_msg and pose_msg")

        # Create Occupancy Grid
        self.info_log.info("Creating occupancy grid")
        occupancy_grid = self.create_occupancy_grid(scan_msg)
        
        # Create Cost Map
        self.info_log.info("Creating cost map")
        try:
            cost_map = self.update_cost_map(occupancy_grid)
        except Exception as e:
            self.warn_log.warn("Error updating cost map, skipping iteration")
            return

        # Create Trajectories
        self.info_log.info("Sampling trajectories")
        trajectories, actions = self.sample_trajectories(self.get_parameter("num_trajectories").value,
                                                         self.get_parameter("steps_trajectories").value)

        # Evaluate Trajectories
        self.info_log.info("Evaluating trajectories")
        min_cost_idx = self.evaluate_trajectories(cost_map, trajectories)

        # # Update u_mean
        # self.u_mean = actions[min_cost_idx, 0]
        self.u_mean[0] = actions[min_cost_idx, 0, 0]

        # Publish AckermannDriveStamped Message
        self.info_log.info("Publishing drive command")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = actions[min_cost_idx, 0, 1]
        drive_msg.drive.speed = actions[min_cost_idx, 0, 0]
        self.drive_pub_.publish(drive_msg)
    
    def create_occupancy_grid(self, scan_msg: LaserScan) -> np.ndarray:
        '''
        Process the LaserScan data into an occupancy grid

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
        Returns:
            occupancy_grid (ndarray): The processed occupancy grid
        '''

        # Initialize an empty occupancy grid (2D array)
        occupancy_grid = np.zeros((self.og.info.width, self.og.info.width), dtype=int)

        # Convert laser scan ranges to grid coordinates
        angles = scan_msg.angle_min + scan_msg.angle_increment * np.arange(len(scan_msg.ranges))
        ranges = np.array(scan_msg.ranges)

        x_coords = np.round((ranges * np.sin(angles)) / self.og.info.resolution + self.og.info.width / 2).astype(int)
        y_coords = np.round((ranges * np.cos(angles)) / self.og.info.resolution).astype(int)

        # Filter out any points that fall outside the grid boundaries
        valid_mask = ((x_coords > 0) & (x_coords < self.og.info.width) &
                      (y_coords > 0) & (y_coords < self.og.info.height))

        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Mark occupied cells in the grid
        occupancy_grid[x_coords, y_coords] = 100

        # Define a simple structuring element (kernel) for dilation
        # dilation_kernel = np.array([[0, 1, 0],
        #                             [1, 1, 1],
        #                             [0, 1, 0]], dtype=bool)

        occupancy_grid[45:55, 0:10] = 0 # TODO: Put this in real world coordinates

        # Apply binary dilation to expand obstacles
        dilation_kernel = np.ones((self.get_parameter("obstacle_dilation").value, self.get_parameter("obstacle_dilation").value), dtype=bool)
        occupancy_grid = binary_dilation(occupancy_grid, structure=dilation_kernel).astype(int) * 100

        # Prepare and publish the updated occupancy grid
        if self.get_parameter("visualize").value:
            self.og.data = occupancy_grid.flatten().tolist()
            self.og.header.stamp = self.get_clock().now().to_msg()
            self.occupancy_pub_.publish(self.og)

        return occupancy_grid
    
    def update_cost_map(self, occupancy_grid: np.ndarray) -> np.ndarray:
        '''
        Update cost map based on current environment.
        We want areas that deviate from the raceline to have higher cost.
        Areas that are closer to obstacles in the occupancy_grid should have higher cost.

        Args:
            occupancy_grid (ndarray): The processed occupancy grid
        Returns:
            cost_map (ndarray): The cost map
        '''

        # Get transform from map to ego frame
        transform = self.tf_buffer.lookup_transform(self.get_parameter("vehicle_frame").value,
                                                    "map", rclpy.time.Time(), timeout=Duration(seconds=0.1))

        # Get vehicle yaw
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        # Determine translation and rotation
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw),  np.cos(yaw)]])
        T = np.array([transform.transform.translation.x,
                      transform.transform.translation.y])

        # Transform all waypoints from map to ego frame
        points_transformed = np.dot(self.waypoints[:, :2], R.T) + T

        # Convert the transformed points into grid indices.
        x_idx = np.round((points_transformed[:, 0] / self.cost_map.info.resolution)).astype(int)
        y_idx = np.round((points_transformed[:, 1] / self.cost_map.info.resolution) + self.cost_map.info.height / 2).astype(int)

        raceline_mask = np.zeros_like(occupancy_grid, dtype=int)
        mask_shape = raceline_mask.shape
        valid = (x_idx >= 0) & (x_idx < mask_shape[0]) & (y_idx >= 0) & (y_idx < mask_shape[1])
        raceline_mask[y_idx[valid], x_idx[valid]] = 100

        # Apply binary dilation to expand raceline
        # dilation_kernel = np.array([[0, 1, 0],
        #                             [1, 1, 1],
        #                             [0, 1, 0]], dtype=bool)
        dilation_kernel = np.ones((self.get_parameter("raceline_dilation").value, self.get_parameter("raceline_dilation").value), dtype=bool)
        raceline_mask = binary_dilation(raceline_mask, structure=dilation_kernel).astype(int) * 100

        # Compute the distance from raceline, for each grid cell
        raceline_cost = self.get_parameter("dist_cost_mult").value * distance_transform_edt(raceline_mask == 0)

        # Final cost map is a weighted sum of the obstacle cost and raceline cost
        cost_map = self.get_parameter("obstacle_weight").value * occupancy_grid + self.get_parameter("raceline_weight").value * raceline_cost

        if self.get_parameter("visualize").value:
            cost_map = np.clip(cost_map, 0, 100).astype(int) # NOTE: Should we be clipping or normalizing? Do we need to?
            self.cost_map.data = cost_map.flatten().tolist()
            self.cost_map.header.stamp = self.get_clock().now().to_msg()
            self.cost_map_pub_.publish(self.cost_map)

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

        # Sample control values
        # v = self.u_mean[0] + np.random.randn(num_trajectories, steps_trajectories - 1, 1) * self.get_parameter("v_sigma").value
        # omega = self.u_mean[1] + np.random.randn(num_trajectories, steps_trajectories - 1, 1) * self.get_parameter("omega_sigma").value

        v = (self.u_mean[0] + np.random.randn(num_trajectories, 1, 1) * self.get_parameter("v_sigma").value)
        omega = (self.u_mean[1] + np.random.randn(num_trajectories, 1, 1) * self.get_parameter("omega_sigma").value)

        v = np.repeat(v, steps_trajectories - 1, axis=1)
        omega = np.repeat(omega, steps_trajectories - 1, axis=1)

        # Limit control values
        v = np.clip(v, self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
        omega = np.clip(omega, -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)

        actions = np.concatenate((v, omega), axis=2)

        # Sample trajectories
        trajectories = np.zeros((num_trajectories, steps_trajectories, 3))
        for i in range(steps_trajectories - 1):
            trajectories[:, i + 1] = self.model.predict_euler(trajectories[:, i], actions[:, i])

        # Publish a subset of trajectories
        self.publish_trajectories(trajectories[:5])

        return trajectories, actions
    
    def evaluate_trajectories(self, cost_map: np.ndarray, trajectories: np.ndarray) -> int:
        '''
        Evaluate trajectories using the cost map

        Args:
            cost_map (ndarray): The cost map
            trajectories (np.ndarray): (num_trajectories x steps_trajectories x 3) Sampled trajectories
        Returns:
            min_cost_idx (int): The index of the trajectory with the lowest cost
        '''

        # Convert each trajectory to the cost map frame
        trajectories_pixels = trajectories / self.cost_map.info.resolution
        trajectories_pixels[:, :, 0] = trajectories_pixels[:, :, 0]
        trajectories_pixels[:, :, 1] = trajectories_pixels[:, :, 1] + self.cost_map.info.height/2

        # Handle trajectories that fall outside of the cost map
        trajectories_pixels = np.clip(trajectories_pixels[:, :, :2], 0, self.cost_map.info.width - 1)

        # Evaluate trajectories and determine the lowest cost
        traj_scores = np.sum(cost_map[trajectories_pixels[:, :, 1].astype(int), trajectories_pixels[:, :, 0].astype(int)], axis=1)

        # now we will compare trajectory headings with waypoint headings
        # Extract trajectory positions
        traj_xy = trajectories[:, :, :2] # shape (N, T, 2)
        # Want shape (N, T, 1, 2) - (1, 1, W, 2) to become (N, T, W, 2)
        diff = traj_xy[:, :, np.newaxis, :] - self.waypoints[np.newaxis, np.newaxis, :, :2]
        dists = np.linalg.norm(diff, axis=-1) 

        # Find the index of the closest waypoint at each step
        closest_wp_indices = np.argmin(dists, axis=-1) # shape is (N,T)
        desired_headings = np.take(self.waypoints[:, 2], closest_wp_indices) # use vectorized indexing
        # Extract the predicted headings
        traj_headings = trajectories[:, :, 2]  # Shape: (N, T)
        # Compute wrapped angular difference for each step.
        heading_errors = np.abs((traj_headings - desired_headings + np.pi) % (2 * np.pi) - np.pi)

        # Compute the per-trajectory heading cost by summing over all T steps.
        heading_cost_per_traj = np.sum(heading_errors * self.get_parameter("heading_weight").value, axis=1)
        traj_scores += heading_cost_per_traj.astype(int)

        min_cost_idx = np.argmin(traj_scores)

        # Publish a lowest cost trajectory
        self.publish_trajectories(np.expand_dims(trajectories[min_cost_idx], 0), np.array([1.0, 0.0, 0.0]))

        return min_cost_idx

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
            marker.header.frame_id = "map"
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