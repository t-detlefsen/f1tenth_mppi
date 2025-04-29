#!/usr/bin/python3

import copy
import math
import numpy as np
from scipy.ndimage import binary_dilation
from typing import Tuple
import torch

import rclpy
import tf2_ros
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
import transforms3d

from f1tenth_mppi.utils import *
from f1tenth_mppi.dynamics_models import KBM

class MPPI(Node):
    def __init__(self):
        super().__init__('MPPI')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.u_prev = torch.zeros((self.get_parameter("steps_trajectories").value, 2), device=self.device)

        # Load Waypoints
        try:
            self.waypoints = torch.from_numpy(load_waypoints(self.get_parameter("waypoint_path").value)).to(self.device)
            self.waypoints[:, 3] = torch.clip(self.waypoints[:, 3], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
            # self.waypoints[:, 3] = (self.waypoints[:, 3]) / np.max(self.waypoints[:, 3]) * self.get_parameter("max_throttle").value
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
        self.marker_pub_ = self.create_publisher(MarkerArray,
                                                self.get_parameter("marker_topic").value,
                                                10)

        # Visualize waypoints
        if self.get_parameter("visualize").value:
            self.publish_markers(self.waypoints, torch.tensor([1.0, 0.0, 0.0]), "waypoints")

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
        self.create_occupancy_grid(scan_msg)

        # ---------- MPPI ----------
        # Set initial state
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        yaw = transforms3d.euler.quat2euler(quat)[0]

        if yaw < 0:
            yaw += 2 * np.pi

        x0 = torch.tensor([pose_msg.pose.pose.position.x,
                       pose_msg.pose.pose.position.y,
                       yaw]).to(self.device)
        
        # Load previous control input sequence
        u = self.u_prev

        # Calculate noise to add to control
        mu = torch.zeros(2).to(self.device)
        sigma = torch.tensor([[self.get_parameter("v_sigma").value, 0.0],
                          [0.0, self.get_parameter("omega_sigma").value]]).to(self.device)
        
        # epsilon shape will be (N, T, 2)
        mvn = torch.distributions.MultivariateNormal(mu, sigma)
        epsilon = mvn.sample((self.get_parameter("num_trajectories").value - 1, self.get_parameter("steps_trajectories").value,))  
        epsilon = torch.cat([torch.zeros(1, self.get_parameter("steps_trajectories").value, 2, device=self.device), epsilon], dim=0)
        # print(f"new epsilon shape is {epsilon.shape}")

        # Initialize state cost
        S = torch.zeros(self.get_parameter("num_trajectories").value).to(self.device)

        v = torch.zeros((self.get_parameter("num_trajectories").value,
                      self.get_parameter("steps_trajectories").value,
                      2)).to(self.device)

        x = torch.zeros((self.get_parameter("num_trajectories").value,
                      self.get_parameter("steps_trajectories").value+1,
                    3)).to(self.device)
        x[:, 0] = x0
        
        # Loop through timesteps
        for j in range(1, self.get_parameter("steps_trajectories").value):
            # Add noise to control
            v[:, j-1] = u[j-1] + epsilon[:, j-1]

            # Clamp control inputs
            v[:, j-1, 0] = torch.clip(v[:, j-1, 0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
            v[:, j-1, 1] = torch.clip(v[:, j-1, 1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)

            # Update state
            x[:, j] = self.model.predict_euler(x[:, j-1], v[:, j-1])

            # Add stage cost
            S += self.compute_cost(x[:, j], v[:, j, 0], pose_msg).squeeze()

        # # Add terminal cost
        S += self.compute_cost(x[:, j], v[:, j, 0], pose_msg).squeeze() * 10

        # Compute information theoretic weights for each sample ???
        w = self.compute_weights(S)
        # Calculate + smooth added noise
        w_epsilon = torch.sum(torch.multiply(w[:, None, None], epsilon), dim=0)

        w_epsilon = self.moving_average(w_epsilon, 4)

        # Update control input sequence
        u += w_epsilon
        # u = v[np.argmin(S)]

        u[:, 0] = torch.clip(u[:, 0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
        u[:, 1] = torch.clip(u[:, 1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)
        # --------------------------

        # Update control sequence
        self.u_prev[:-1] = u[1:].clone()
        self.u_prev[-1] = u[-1].clone()
            
        # ALL TRAJECTORIES
        trajs = torch.zeros((self.get_parameter("num_trajectories").value, self.get_parameter("steps_trajectories").value+1, 3)).to(self.device)
        trajs[:, 0] = x0
        for i in range(self.get_parameter("steps_trajectories").value):
            trajs[:, i+1] = self.model.predict_euler(trajs[:, i], v[:, i])
        self.publish_trajectories(trajs[:10], ns="all")

        # OPTIMAL TRAJECTORY
        traj = torch.zeros((self.get_parameter("steps_trajectories").value+1, 3)).to(self.device)
        traj[0] = x0
        for i in range(self.get_parameter("steps_trajectories").value):
            traj[i+1] = self.model.predict_euler(torch.unsqueeze(traj[i], 0),
                                         torch.unsqueeze(u[i], 0)).squeeze()
        self.publish_trajectories(torch.unsqueeze(traj, 0), color=torch.tensor([1.0, 0.0, 0.0]), ns="opt")

        # Publish AckermannDriveStamped Message
        self.info_log.info("Publishing drive command")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = u[0, 0].item()
        drive_msg.drive.steering_angle = u[0, 1].item()
        self.drive_pub_.publish(drive_msg)

        # print(f"Speed is {u[0, 0].item()} and steering angle is {u[0, 1].item()}")
        # exit()

    def compute_cost(self, x_t: np.ndarray, v_t: np.ndarray, pose_msg: Odometry) -> np.ndarray:
        '''
        Calculate stage cost

        Args:
            x_t (np.ndarray): current state
            pose_msg (Odometry): Robot pose
        Returns:
            stage_cost (float): computed stage cost
        '''
        # print(f"type of x_t is {type(x_t)}")
        stage_cost_weights = [13.5, 13.5, 5.5, 5.5] # TODO: add these to params.yaml
        x, y, yaw = torch.hsplit(x_t, 3)
        # print(f"x/y/yaw shape is {x.shape, y.shape, yaw.shape}")
        # v = np.expand_dims(v_t, 1)
        v = torch.unsqueeze(v_t, 1)
        yaw[yaw < 0] = yaw[yaw < 0] + 2 * np.pi

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        
        # Fix yaw
        ref_yaw = ref_yaw + np.pi / 2
        ref_yaw[ref_yaw < 0] = ref_yaw[ref_yaw < 0] + 2 * np.pi
        ref_yaw[ref_yaw - yaw > 4.5] = torch.abs(
            ref_yaw[ref_yaw - yaw > 4.5] - (2 * np.pi)
        )
        ref_yaw[ref_yaw - yaw < -4.5] = torch.abs(
            ref_yaw[ref_yaw - yaw < -4.5] + (2 * np.pi)
        )

        stage_cost = stage_cost_weights[0]*(x-ref_x)**2 + stage_cost_weights[1]*(y-ref_y)**2 + \
                     stage_cost_weights[2]*(ref_yaw-ref_yaw)**2 + stage_cost_weights[3]*(v-ref_v)**2

        # add penalty for collision with obstacles
        # stage_cost += np.expand_dims(self.is_collided(x_t, pose_msg), 1) * 1.0e10
        stage_cost += torch.unsqueeze(self.is_collided(x_t, pose_msg), 1) * 1.0e10

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
        cur_state = torch.hstack((x,y))

        # print(f"Shape of cur_state is {cur_state.shape}")

        # Repeat for distance calculation (Shape == num_trajectories x len_waypoints x 2)
        cur_state_repeat = torch.repeat_interleave(torch.unsqueeze(cur_state, 1), len(self.waypoints), dim=1)
        waypoints_repeat = torch.repeat_interleave(torch.unsqueeze(self.waypoints[:, :2], 0), self.get_parameter("num_trajectories").value, dim=0)
        distances = torch.norm(cur_state_repeat - waypoints_repeat, dim=2)
        min_idx = torch.argmin(distances, dim=1)
        wp_x, wp_y, wp_yaw, wp_v = torch.hsplit(self.waypoints[min_idx], 4)
        return min_idx, wp_x, wp_y, wp_yaw, wp_v

    def is_collided(self, x_t: np.ndarray, pose_msg: Odometry) -> np.ndarray:
        '''
        Checks if the current state is collided with an obstacle

        Args:
            x_t (np.ndarray): current state
            pose_msg (Odometry): Received pose odometry message
        Returns:
            is_collided (bool): Returns whether or not the input state is on an obstacle
        '''
        occupancy_data = np.array(self.og.data).reshape((self.og.info.height, self.og.info.width))
        occupancy_data = torch.from_numpy(occupancy_data).to(self.device)
        # print(f"occupancy data in gpu is shape {occupancy_data.shape}")

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        # yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        R10 = torch.tensor([2 * (qw * qz + qx * qy)])
        R00 = torch.tensor([1 - 2 * (qy**2 + qz**2)])
        yaw = torch.atan2(R10, R00)

        R = torch.tensor([[torch.cos(yaw), -torch.sin(yaw)],
                    [torch.sin(yaw),  torch.cos(yaw)]]).to(self.device)
        T = torch.tensor([pose_msg.pose.pose.position.x,
                      pose_msg.pose.pose.position.y]).to(self.device)
        # import ipdb; ipdb.set_trace()
        # convert the state into egocar frame, then occupancy grid frame
        state_pos = torch.matmul(x_t[:, :2] - T, R)
        
        state_pos_pixels = state_pos / self.og.info.resolution 
        state_pos_pixels[:, 1] = state_pos_pixels[:, 1] + (self.og.info.height / 2)
        state_pos_pixels = torch.clip(state_pos_pixels, 0, self.og.info.width - 1)

        return occupancy_data[state_pos_pixels[:, 1].to(torch.int32), state_pos_pixels[:, 0].to(torch.int32)] > 0

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

        param_lambda = 0.1 # TODO: Make parameter

        # Calculate eta
        eta = torch.sum(torch.exp((-1.0/param_lambda) * (S-rho)))

        # Calculate weight
        w = (1.0 / eta) * torch.exp( (-1.0/param_lambda) * (S-rho) )

        return w

    def moving_average(self, xx: torch.Tensor, window_size: int) -> torch.Tensor:
        '''
        Apply moving average filter to sequence

        Args:
            xx (Tensor): Sequence, shape [T, D], on self.device
            window_size (int): Size of window to apply filter
        Returns:
            xx_mean (Tensor): Smoothed sequence, shape [T, D]
        '''

        # build your box‐filter kernel once
        b = torch.ones(window_size, device=self.device) / window_size
        kernel = b.view(1, 1, window_size)         # shape [out_channels=1, in_channels=1, K]
        pad = window_size // 2

        T, D = xx.shape
        xx_mean = torch.zeros_like(xx, device=self.device)

        for d in range(D):
            # grab one channel, make it [1,1,T]
            s = xx[:, d].unsqueeze(0).unsqueeze(0)      # shape [1,1,T]
            # asymmetric pad so output stays T
            pad_left  = window_size // 2
            pad_right = window_size - 1 - pad_left
            s_padded  = torch.nn.functional.pad(s, (pad_left, pad_right), mode='constant', value=0)

            y = torch.nn.functional.conv1d(s_padded, kernel, padding=0)  # [1,1,T]

            # # conv1d does cross-correlation, which for a symmetric box filter is the same as convolution
            xx_mean[:, d] = y.view(T)                  # back to [T]

            # now your existing edge‐corrections
            n_conv = math.ceil(window_size/2)
            xx_mean[0, d] *= window_size / n_conv
            for i in range(1, n_conv):
                xx_mean[i,   d] *= window_size / (i + n_conv)
                xx_mean[-i,  d] *= window_size / (i + n_conv - (window_size % 2))
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
        occupancy_grid = torch.zeros((self.og.info.width, self.og.info.width), dtype=int, device=self.device)

        # Convert laser scan ranges to grid coordinates
        angles = scan_msg.angle_min + scan_msg.angle_increment * torch.arange(len(scan_msg.ranges))
        ranges = torch.tensor(scan_msg.ranges).to(self.device)

        # print(f"Ranges shape is {ranges.shape} and occ grid shape is {occupancy_grid.shape}")

        x_coords = torch.round((ranges * torch.sin(angles)) / self.og.info.resolution + self.og.info.width / 2).to(torch.int32)
        y_coords = torch.round((ranges * torch.cos(angles)) / self.og.info.resolution).to(torch.int32)

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
        dilation_kernel = torch.ones((self.get_parameter("occupancy_dilation").value, self.get_parameter("occupancy_dilation").value), dtype=torch.bool, device=self.device)
        occupancy_grid = torch.from_numpy(binary_dilation(occupancy_grid, structure=dilation_kernel)).to(torch.int32) * 100

        # Prepare and publish the updated occupancy grid
        self.og.data = occupancy_grid.flatten().tolist()
        if self.get_parameter("visualize").value:
            self.og.header.stamp = self.get_clock().now().to_msg()
            self.occupancy_pub_.publish(self.og)

        # print(f"Occupancy grid type is {type(occupancy_grid)} and shape is {occupancy_grid.shape}")
        # print(f"min and max of occ grid is {occupancy_grid.min(), occupancy_grid.max()}")

        return occupancy_grid
    
    def publish_trajectories(self, points: np.ndarray, color: np.ndarray = np.array([0.0, 0.0, 1.0]), ns: str = ""):
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
            for i in range(0, points.shape[1] - 1, 2):
                marker = Marker()
                marker.ns = ns
                marker.header.frame_id = "map"
                marker.id = j * points.shape[1] + i
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD
                marker.color.r = color[0].item()
                marker.color.g = color[1].item()
                marker.color.b = color[2].item()
                marker.color.a = 1.0
                marker.scale.x = 0.01

                p1 = Point()
                p1.x = points[j, i, 0].item()
                p1.y = points[j, i, 1].item()
                p2 = Point()
                p2.x = points[j, i+1, 0].item()
                p2.y = points[j, i+1, 1].item()

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
            marker.pose.position.x = points[i, 0].item()
            marker.pose.position.y = points[i, 1].item()
            marker.color.r = color[0].item()
            marker.color.g = color[1].item()
            marker.color.b = color[2].item()
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