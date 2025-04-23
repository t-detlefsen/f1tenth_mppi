#!/usr/bin/python3

import copy
import math
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation
from typing import Tuple

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
        yaw = transforms3d.euler.quat2euler(quat)[0]

        if yaw < 0:
            yaw += 2 * np.pi

        x0 = np.array([pose_msg.pose.pose.position.x,
                       pose_msg.pose.pose.position.y,
                       yaw]) # NOTE: Make sure yaw is correct
        
        # Load previous control input sequence
        u = self.u_prev
        
        # TODO: Find the nearest waypoint
        # NOTE Brian: don't think we need this, in the original code they update the nearest waypoint for computation speedup. but our nearest waypoint code is vectorized hahaha

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
                x = x.squeeze()

                # Add stage cost
                S[i] += self.compute_stage_cost(x, pose_msg)

            # Add terminal cost
            S[i] += self.compute_terminal_cost(x, pose_msg)

        # Compute information theoretic weights for each sample ???
        w = self.compute_weights(S)

        # Calculate + smooth added noise
        w_epsilon = np.zeros((self.get_parameter("steps_trajectories").value, 2))
        for i in range(self.get_parameter("steps_trajectories").value):
            for j in range(self.get_parameter("num_trajectories").value):
                w_epsilon[i] += w[j] * epsilon[j, i]

        w_epsilon = self.moving_average(w_epsilon, 10)

        # Update control input sequence
        u += w_epsilon

        u[:, 0] = np.clip(u[:, 0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
        u[:, 1] = np.clip(u[:, 1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)
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

    def compute_stage_cost(self, x_t: np.ndarray, pose_msg: Odometry) -> float:
        '''
        Calculate stage cost

        Args:
            x_t (np.ndarray): current state
            pose_msg (Odometry): Robot pose
        Returns:
            stage_cost (float): computed stage cost
        '''
        stage_cost_weights = [50.0, 50.0, 1.0] # TODO: add these to params.yaml
        x, y, yaw = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, _ = self.get_nearest_waypoint(x, y)
        
        # Fix yaw
        ref_yaw = ref_yaw + np.pi / 2
        ref_yaw = ((ref_yaw + 2.0*np.pi) % (2.0*np.pi))
        if ref_yaw - yaw > 4.5:
            ref_yaw = abs(ref_yaw - 2 * np.pi)
        elif ref_yaw - yaw < -4.5:
            ref_yaw = abs(ref_yaw + 2 * np.pi)

        stage_cost = stage_cost_weights[0]*(x-ref_x)**2 + stage_cost_weights[1]*(y-ref_y)**2 + stage_cost_weights[2]*(ref_yaw-ref_yaw)**2 

        # print(f"Vehicle x: {x} Traj x: {ref_x}")
        # print(f"Vehicle y: {y} Traj y: {ref_y}")
        # print(f"Vehicle yaw: {yaw} Traj yaw: {ref_yaw}")

        # add penalty for collision with obstacles
        stage_cost += self.is_collided(x_t, pose_msg) * 1.0e10

        return stage_cost

    def compute_terminal_cost(self, x_T: np.ndarray, pose_msg: Odometry) -> float:
        '''
        Calculate terminal cost

        Args:
            x_T (np.ndarray): final state
            pose_msg (Odometry): Robot pose
        Returns:
            terminal_cost (float): computed terminal cost
        '''
        terminal_cost_weights = [50.0, 50.0, 1.0] # TODO: add these to params.yaml
        x, y, yaw = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, _ = self.get_nearest_waypoint(x, y)
        ref_yaw = ref_yaw + np.pi / 2
        stage_cost = terminal_cost_weights[0]*(x-ref_x)**2 + terminal_cost_weights[1]*(y-ref_y)**2 + terminal_cost_weights[2]*(yaw-ref_yaw)**2 
        
        # add penalty for collision with obstacles
        stage_cost += self.is_collided(x_T, pose_msg) * 1.0e10

        return stage_cost

    def get_nearest_waypoint(self, x: float, y: float) -> Tuple[float, float, float, float, float]:
        '''
        Search the closest waypoint to the vehicle on the reference path

        Args:
            x (float): Input x position
            y (float): Input y position
        Returns:
            nearest_waypoint (Tuple): Returns waypoint index and information
        '''
        cur_state = np.array([x, y]) 
        distances = np.linalg.norm(cur_state - self.waypoints[:, :2], axis=1) # shape is (N,)
        min_idx = np.argmin(distances)
        wp_x, wp_y, wp_yaw, wp_v = self.waypoints[min_idx]
        return min_idx, wp_x, wp_y, wp_yaw, wp_v
    
    def is_collided(self, x_t: np.ndarray, pose_msg: Odometry) -> bool:
        '''
        Checks if the current state is collided with an obstacle

        Args:
            x_t (np.ndarray): current state
            pose_msg (Odometry): Received pose odometry message
        Returns:
            is_collided (bool): Returns whether or not the input state is on an obstacle
        '''
        occupancy_data = np.array(self.og.data).reshape((self.og.info.height, self.og.info.width))

        car_x, car_y = pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y

        # convert the state into egocar frame, then occupancy grid frame
        state_pos = np.array([x_t[0], x_t[1]])      
        state_pos[0] -= car_x
        state_pos[1] -= car_y
        state_pos_pixels = state_pos / self.og.info.resolution
        state_pos_pixels[1] = state_pos_pixels[1] + (self.og.info.height / 2)
        state_pos_pixels = np.clip(state_pos_pixels, 0, self.og.info.width - 1)

        if occupancy_data[state_pos_pixels[1].astype(int), state_pos_pixels[0].astype(int)] > 0:
            return True
        
        return False

    def compute_weights(self, S: np.ndarray) -> np.ndarray:
        '''
        Compute information theoretic weights for each sample

        Args:
            S (ndarray): Cost of each trajectory
        Returns:
            w (ndarray): Weight for each trajectory
        '''
        # Calculate rho
        rho = S.min()

        param_lambda = 100.0 # TODO: Make parameter

        # Calculate eta
        eta = 0.0
        for k in range(self.get_parameter("num_trajectories").value):
            eta += np.exp((-1.0/param_lambda) * (S[k]-rho))

        # Calculate weight
        w = np.zeros(self.get_parameter("num_trajectories").value)
        for k in range(self.get_parameter("num_trajectories").value):
            w[k] = (1.0 / eta) * np.exp( (-1.0/param_lambda) * (S[k]-rho) )
        
        return w

    def moving_average(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        '''
        Apply moving average filter to sequence

        Args:
            xx (ndarray): Sequence
            window_size (ndarray): Size of window to apply filter
        Returns:
            xx_mean (ndarray): Smoothed sequence
        '''

        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean

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
        self.og.data = occupancy_grid.flatten().tolist()
        if self.get_parameter("visualize").value:
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