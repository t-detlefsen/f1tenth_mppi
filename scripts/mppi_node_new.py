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
from transforms3d.euler import quat2euler
import transforms3d

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

        # Initialize Occupancy Grid
        self.og = OccupancyGrid() # Create your occupancy grid here
        self.og.header.frame_id = self.get_parameter("vehicle_frame").value
        self.og.info.resolution = self.get_parameter("cost_map_res").value
        self.og.info.width = self.get_parameter("cost_map_width").value
        self.og.info.height = self.get_parameter("cost_map_width").value
        self.og.info.origin.position.x = 0.0
        self.og.info.origin.position.y = -(self.og.info.height * self.og.info.resolution) / 2

        # Initialize previous control input sequence
        self.u_prev = np.zeros((self.get_parameter("steps_trajectories").value, 2))

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
                ('occupancy_dilation', None),
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

        # ---------- MPPI ----------
        # Set initial state
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]

        x0 = np.array([pose_msg.pose.pose.position.x,
                       pose_msg.pose.pose.position.y,
                       transforms3d.euler.quat2euler(quat)[0]]) # NOTE: Make sure yaw is correct
        
        # Load previous control input sequence
        u = self.u_prev
        
        # TODO: Find the nearest waypoint

        # Calculate noise to add to control
        mu = np.zeros(2)
        sigma = np.array([[self.get_parameter("v_sigma").value, 0.0],
                          [0.0, self.get_parameter("omega_sigma").value]])
        epsilon = np.random.multivariate_normal(mu, sigma, (self.get_parameter("num_trajectories").value, self.get_parameter("steps_trajectories").value))

        # Initialize state cost
        S = np.zeros(self.get_parameter("num_trajectories").value)

        # Loop through trajectories # NOTE: This outer loop can be vectorized
        for i in range(self.get_parameter("num_trajectories").value):
            # Reset state
            x = x0

            # Loop through timesteps
            for j in range(1, self.get_parameter("steps_trajectories").value + 1):
                # Sample control
                if i < 0.95 * self.get_parameter("num_trajectories").value: # TODO: Make parameter
                    u_step = u[j-1] + epsilon[i, j-1] # Exploitation (Add noise to control)
                else:
                    u_step = epsilon[i, j-1] # Exploration (control is noise)

                # Clamp control inputs
                u_step[0] = np.clip(u_step[0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
                u_step[1] = np.clip(u_step[1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)

                # Update state
                x = self.model.predict_euler(np.expand_dims(x, 0),
                                             np.expand_dims(u_step, 0))

                # TODO: Add stage cost
                S[i] += ...

            # TODO: Add terminal cost
            S[i] += ...

        # TODO: Compute information theoretic weights for each sample ???
        # TODO: Calculate + smooth added noise
        # --------------------------

        # Update control sequence
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # Publish AckermannDriveStamped Message
        self.info_log.info("Publishing drive command")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = u[0, 0]
        drive_msg.drive.steering_angle = u[0, 1]
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

        # Ignore wires in lidar sweep
        occupancy_grid[45:55, 0:10] = 0 # TODO: Put this in real world coordinates

        # Apply binary dilation to expand obstacles
        dilation_kernel = np.ones((self.get_parameter("occupancy_dilation").value, self.get_parameter("occupancy_dilation").value), dtype=bool)
        occupancy_grid = binary_dilation(occupancy_grid, structure=dilation_kernel).astype(int) * 100

        # Prepare and publish the updated occupancy grid
        if self.get_parameter("visualize").value:
            self.og.data = occupancy_grid.flatten().tolist()
            self.og.header.stamp = self.get_clock().now().to_msg()
            self.occupancy_pub_.publish(self.og)

        return occupancy_grid

def main(args=None):
    rclpy.init(args=args)
    mppi_node = MPPI()
    rclpy.spin(mppi_node)

    mppi_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()