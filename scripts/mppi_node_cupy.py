#!/usr/bin/python3

import copy
import math
import time
import numpy as np
from scipy.ndimage import binary_dilation
from typing import Tuple
import cupy as cp

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
from f1tenth_mppi.dynamics_models_cupy import KBM

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
        self.u_prev = cp.zeros((self.get_parameter("steps_trajectories").value, 2))

        # Load Waypoints
        try:
            self.waypoints = cp.asarray(load_waypoints(self.get_parameter("waypoint_path").value))
            # self.waypoints[:, 3] = cp.clip(self.waypoints[:, 3], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
            min_v = cp.min(self.waypoints[:, 3]); max_v = max(self.waypoints[:, 3])
            self.waypoints[:, 3] = (self.waypoints[:, 3] - min_v) / (max_v - min_v)
            self.waypoints[:, 3] = self.waypoints[:, 3] * (self.get_parameter("max_throttle").value - self.get_parameter("min_throttle").value) + self.get_parameter("min_throttle").value
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
            self.publish_markers(self.waypoints, np.array([1.0, 0.0, 0.0]), "waypoints")

        # Create message_filters Subscribers
        self.scan_sub_ = Subscriber(self, LaserScan, self.get_parameter("scan_topic").value)
        self.pose_sub_ = Subscriber(self, Odometry, self.get_parameter("pose_topic").value)

        # Create time synchronized callback
        self.time_sync = ApproximateTimeSynchronizer([self.scan_sub_, self.pose_sub_], 10, 0.1)
        self.time_sync.registerCallback(self.callback)

        # num_cuda = cp.cuda.runtime.getDeviceCount()
        # devices = []
        # for i in range(0, num_cuda):
        #     devices += [f'cuda:{i}']
        # print(f"Devices are {devices}")

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
                ('lambda', None),
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
        
        # time.sleep(0.1)
        # self.info_log.info("Recieved scan_msg and pose_msg")
        # drive_msg = AckermannDriveStamped()
        # drive_msg.drive.speed = 0.0
        # drive_msg.drive.steering_angle = self.u_prev[0, 1]
        # self.drive_pub_.publish(drive_msg)

        # Create Occupancy Grid
        self.info_log.info("Creating occupancy grid")

        # start = time.time()
        self.create_occupancy_grid(scan_msg)
        # end = time.time()
        # print(f"Occupancy grid creation took {end - start} sec")

        # ---------- MPPI ----------
        # Set initial state
        quat = [pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z, pose_msg.pose.pose.orientation.w]
        yaw = transforms3d.euler.quat2euler(quat)[0]

        if yaw < 0:
            yaw += 2 * cp.pi

        x0 = cp.array([pose_msg.pose.pose.position.x,
                       pose_msg.pose.pose.position.y,
                       yaw])
        
        # Load previous control input sequence
        u = self.u_prev

        # start = time.time()

        # Calculate noise to add to control
        mu = cp.zeros(2)
        sigma = cp.array([[self.get_parameter("v_sigma").value, 0.0],
                          [0.0, self.get_parameter("omega_sigma").value]])

        epsilon = cp.random.multivariate_normal(mu, sigma, (self.get_parameter("num_trajectories").value - 1, self.get_parameter("steps_trajectories").value))
        epsilon = cp.vstack((cp.zeros((1, self.get_parameter("steps_trajectories").value, 2)), epsilon))
        # end = time.time()
        # print(f"Calculating noise took {end - start} sec")

        # Initialize state cost
        S = cp.zeros(self.get_parameter("num_trajectories").value)

        v = cp.zeros((self.get_parameter("num_trajectories").value,
                      self.get_parameter("steps_trajectories").value,
                      2))

        x = cp.zeros((self.get_parameter("num_trajectories").value,
                      self.get_parameter("steps_trajectories").value+1,
                    3))
        x[:, 0] = x0
        
        # Loop through timesteps
        # start_loop = time.time()
        for j in range(1, self.get_parameter("steps_trajectories").value):
            # start_iter = time.time()
            # Add noise to control
            v[:, j-1] = u[j-1] + epsilon[:, j-1]

            # Clamp control inputs
            v[:, j-1, 0] = cp.clip(v[:, j-1, 0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
            v[:, j-1, 1] = cp.clip(v[:, j-1, 1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)

            # Update noise w/ clamping
            epsilon[:, j-1] = v[:, j-1] - u[j-1]

            # Update state
            x[:, j] = self.model.predict_euler(x[:, j-1], v[:, j-1])

            # Add stage cost
            # param_gamma = 100 * (1.0 - 0.98)
            # correction = u[0] @ cp.linalg.inv(sigma) @ cp.transpose(v[:, 0])
            S += self.compute_cost(x[:, j], v[:, j-1, 0], pose_msg).squeeze()# + param_gamma * correction
            
            # # Different formulation
            # S += (v[:, j, 0] ** 2) * 0.01 + (v[:, j, 0] ** 2) * 0.75 + \
            #      self.compute_cost(x[:, j], v[:, j, 0], pose_msg).squeeze()
                #  ((v[:, j, 0] - v[:, j - 1, 0]) ** 2) * 0.01 + ((v[:, j, 1] - v[:, j - 1, 1]) ** 2) * 100
        #     end_iter = time.time()
        #     if j == 1: print(f"One iteration of the loop took {end_iter - start_iter} sec")
        # end_loop = time.time()
        # print(f"Looping through time steps took {end_loop - start_loop} sec")
        # Add terminal cost
        S += self.compute_cost(x[:, j], v[:, j-1, 0], pose_msg).squeeze() * 10

        # Compute information theoretic weights for each sample ???
        # start_weight = time.time()
        w = self.compute_weights(S)


        # Calculate + smooth added noise
        w_epsilon = cp.sum(cp.multiply(w[:, cp.newaxis, cp.newaxis], epsilon), axis=0)
        # end_weight = time.time()
        # print(f"Computing the weights took {end_weight - start_weight} sec")

        # start_avg = time.time()
        w_epsilon = self.moving_average(w_epsilon, 4)
        # end_avg = time.time()
        # print(f"Computing the avg took {end_avg - start_avg} sec")

        # Update control input sequence
        u += w_epsilon
        # u = v[cp.argmin(S)]

        u[:, 0] = cp.clip(u[:, 0], self.get_parameter("min_throttle").value, self.get_parameter("max_throttle").value)
        u[:, 1] = cp.clip(u[:, 1], -self.get_parameter("max_steer").value, self.get_parameter("max_steer").value)
        # --------------------------

        # Update control sequence
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
            
        # # ALL TRAJECTORIES
        # trajs = cp.zeros((self.get_parameter("num_trajectories").value, self.get_parameter("steps_trajectories").value+1, 3))
        # trajs[:, 0] = x0
        # for i in range(self.get_parameter("steps_trajectories").value):
        #     trajs[:, i+1] = self.model.predict_euler(trajs[:, i], v[:, i])
        # self.publish_trajectories(trajs[:10], ns="all")
        # # print(S)

        # OPTIMAL TRAJECTORY
        # start = time.time()
        traj = cp.zeros((self.get_parameter("steps_trajectories").value+1, 3))
        traj[0] = x0
        for i in range(self.get_parameter("steps_trajectories").value):
            traj[i+1] = self.model.predict_euler(cp.expand_dims(traj[i], 0),
                                         cp.expand_dims(u[i], 0)).squeeze()
        self.publish_trajectories(cp.expand_dims(traj, 0), color=np.array([1.0, 0.0, 0.0]), ns="opt")

        # Publish AckermannDriveStamped Message
        self.info_log.info("Publishing drive command")
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = u[0, 0].item()
        drive_msg.drive.steering_angle = u[0, 1].item()
        self.drive_pub_.publish(drive_msg)

        # end = time.time()
        # print(f"Computing traj and publishing took {end - start} sec")
        # print(f"Speed is {u[0, 0].item()} and steering angle is {u[0, 1].item()}")

    def compute_cost(self, x_t: cp.ndarray, v_t: cp.ndarray, pose_msg: Odometry) -> cp.ndarray:
        '''
        Calculate stage cost

        Args:
            x_t (cp.ndarray): current state
            pose_msg (Odometry): Robot pose
        Returns:
            stage_cost (float): computed stage cost
        '''

        stage_cost_weights = [13.5, 13.5, 5.5, 5.0] # TODO: add these to params.yaml
        x, y, yaw = cp.hsplit(x_t, 3)
        v = cp.expand_dims(v_t, 1)
        yaw[yaw < 0] = yaw[yaw < 0] + 2 * cp.pi

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, ref_v = self.get_nearest_waypoint(x, y)
        
        # Fix yaw
        ref_yaw = ref_yaw + cp.pi / 2
        ref_yaw[ref_yaw < 0] = ref_yaw[ref_yaw < 0] + 2 * cp.pi
        ref_yaw[ref_yaw - yaw > 4.5] = cp.abs(
            ref_yaw[ref_yaw - yaw > 4.5] - (2 * cp.pi)
        )
        ref_yaw[ref_yaw - yaw < -4.5] = cp.abs(
            ref_yaw[ref_yaw - yaw < -4.5] + (2 * cp.pi)
        )

        stage_cost = stage_cost_weights[0]*(x-ref_x)**2 + stage_cost_weights[1]*(y-ref_y)**2 + \
                     stage_cost_weights[2]*(yaw-ref_yaw)**2 + stage_cost_weights[3]*(v-ref_v)**2

        # add penalty for collision with obstacles
        stage_cost += cp.expand_dims(self.is_collided(x_t, pose_msg), 1) * 1.0e10

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
        cur_state = cp.hstack((x,y))

        # Repeat for distance calculation (Shape == num_trajectories x len_waypoints x 2)
        cur_state_repeat = cp.repeat(cp.expand_dims(cur_state, 1), len(self.waypoints), axis=1)
        waypoints_repeat = cp.repeat(cp.expand_dims(self.waypoints[:, :2], 0), self.get_parameter("num_trajectories").value, axis=0)
        distances = cp.linalg.norm(cur_state_repeat - waypoints_repeat, axis=2)
        min_idx = cp.argmin(distances, axis=1)
        wp_x, wp_y, wp_yaw, wp_v = cp.hsplit(self.waypoints[min_idx], 4)
        return min_idx, wp_x, wp_y, wp_yaw, wp_v

    def is_collided(self, x_t: cp.ndarray, pose_msg: Odometry) -> cp.ndarray:
        '''
        Checks if the current state is collided with an obstacle

        Args:
            x_t (cp.ndarray): current state
            pose_msg (Odometry): Received pose odometry message
        Returns:
            is_collided (bool): Returns whether or not the input state is on an obstacle
        '''
        occupancy_data = cp.array(self.og.data).reshape((self.og.info.height, self.og.info.width))

        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        yaw = cp.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

        R = cp.array([[cp.cos(yaw), -cp.sin(yaw)],
                    [cp.sin(yaw),  cp.cos(yaw)]])
        T = cp.array([pose_msg.pose.pose.position.x,
                      pose_msg.pose.pose.position.y])

        # convert the state into egocar frame, then occupancy grid frame
        state_pos = cp.dot(x_t[:, :2] - T, R)
        
        state_pos_pixels = state_pos / self.og.info.resolution
        state_pos_pixels[:, 1] = state_pos_pixels[:, 1] + (self.og.info.height / 2)
        state_pos_pixels = cp.clip(state_pos_pixels, 0, self.og.info.width - 1)

        return occupancy_data[state_pos_pixels[:, 1].astype(int), state_pos_pixels[:, 0].astype(int)] > 0

    def compute_weights(self, S: cp.ndarray) -> cp.ndarray:
        '''
        Compute information theoretic weights for each sample

        Args:
            S (ndarray): Cost of each trajectory
        Returns:
            w (ndarray): Weight for each trajectory
        '''
        # Calculate rho
        rho = S.min()

        # Calculate eta
        eta = cp.sum(cp.exp((-1.0/self.get_parameter("lambda").value) * (S-rho)))

        # Calculate weight
        w = (1.0 / eta) * cp.exp( (-1.0/self.get_parameter("lambda").value) * (S-rho) )

        return w

    def moving_average(self, xx: cp.ndarray, window_size: int) -> cp.ndarray:
        '''
        Apply moving average filter to sequence

        Args:
            xx (ndarray): Sequence
            window_size (ndarray): Size of window to apply filter
        Returns:
            xx_mean (ndarray): Smoothed sequence
        '''

        """apply moving average filter for smoothing icput sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control icputs.
        """
        b = cp.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = cp.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = cp.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean

    def create_occupancy_grid(self, scan_msg: LaserScan) -> cp.ndarray:
        '''
        Process the LaserScan data into an occupancy grid

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
        Returns:
            occupancy_grid (ndarray): The processed occupancy grid
        '''

        # Initialize an empty occupancy grid (2D array)
        occupancy_grid = cp.zeros((self.og.info.width, self.og.info.width), dtype=int)

        # Convert laser scan ranges to grid coordinates
        angles = scan_msg.angle_min + scan_msg.angle_increment * cp.arange(len(scan_msg.ranges))
        ranges = cp.array(scan_msg.ranges)

        x_coords = cp.round((ranges * cp.sin(angles)) / self.og.info.resolution + self.og.info.width / 2).astype(int)
        y_coords = cp.round((ranges * cp.cos(angles)) / self.og.info.resolution).astype(int)

        # Filter out any points that fall outside the grid boundaries
        valid_mask = ((x_coords > 0) & (x_coords < self.og.info.width) &
                      (y_coords > 0) & (y_coords < self.og.info.height))

        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]

        # Mark occupied cells in the grid
        occupancy_grid[x_coords, y_coords] = 100

        # # Ignore wires in lidar sweep
        # occupancy_grid[45:55, 0:10] = 0 # TODO: Put this in real world coordinates

        # Apply binary dilation to expand obstacles
        dilation_kernel = cp.ones((self.get_parameter("occupancy_dilation").value, self.get_parameter("occupancy_dilation").value), dtype=bool)
        occupancy_grid = binary_dilation(occupancy_grid.get(), structure=dilation_kernel.get()).astype(int) * 100

        # Prepare and publish the updated occupancy grid
        self.og.data = occupancy_grid.flatten().tolist()
        if self.get_parameter("visualize").value:
            self.og.header.stamp = self.get_clock().now().to_msg()
            self.occupancy_pub_.publish(self.og)

        return occupancy_grid
    
    def publish_trajectories(self, points: cp.ndarray, color: cp.ndarray = np.array([0.0, 0.0, 1.0]), ns: str = ""):
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
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
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
    
    def publish_markers(self, points: cp.ndarray, color: cp.ndarray = np.array([1.0, 0.0, 0.0]), ns: str = ""):
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