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

        self.temp_time = 0
        self.oc_pub_ = self.create_publisher(OccupancyGrid, "ego_occupancy", 10) # and other publishers that you might need
                
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
            ])
        
    def transform_point(self, point):
        '''
        Transform the point from global map frame to local egocar frame

        Args:
            point (Tuple): The point in ego frame
        Returns:
            point (Tuple): The point in map frame
        '''
        point_stamped = PointStamped()
        point_stamped.header.frame_id = 'map'
        point_stamped.point = Point(x=point[0], y=point[1], z=0.0)
        try:
            transformed_point = self.tf_buffer.transform(point_stamped, "ego_racecar/base_link")
        except Exception as e:
            print(f"Got Exception: {e}")
            return
        return [transformed_point.point.x, transformed_point.point.y]

    def callback(self, scan_msg: LaserScan, pose_msg: Odometry):
        '''
        Time synchronized callback to handle LaserScan and Odometry messages

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
            pose_msg (Odometry): Odometry message from the particle filter
        '''

        self.info_log.info("Recieved scan_msg and pose_msg")

        # Get the transformation from map frame to local egocar frame
        try:
            self.tf_buffer.lookup_transform("ego_racecar/base_link", "map", rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception as e:
            print(f"not working here")
            print(e)
            return

        # TODO: Update parameters

        # TODO: Create Occupancy Grid
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

        # Prepare and publish the updated occupancy grid
        self.og.data = grid.flatten().tolist()
        self.og.header.stamp = self.get_clock().now().to_msg()
        self.oc_pub_.publish(self.og)
        
        # TODO: Create Cost Map
        self.info_log.info("Creating cost map")
        ego_position = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y)
        occupancy_grid = None # TODO: Update this once create_occupancy_grid() is completed
        self.update_cost_map(ego_position, occupancy_grid)

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
        # occupancy_data = np.array(occupancy_grid.data).reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        occupancy_data = np.zeros((100, 100))
        obstacles = occupancy_data > 0
        distance_from_obstacle = distance_transform_edt(~obstacles)
        obstacle_cost = 1.0 / (distance_from_obstacle + 1e-9)

        # Compute the Raceline Deviation Cost Component
        raceline_mask = np.zeros_like(occupancy_data, dtype=bool)
        # Convert raceline waypoints into egocar frame and into grid indices
        waypoints = self.waypoints[:, :2]
        for point in waypoints:
            # print(f"waypoint is {point}")
            new_point = self.transform_point(point)
            # print(f"new point is {new_point}")
            # Cast to integer indices (using int() may be replaced by proper rounding or transformation)
            x_idx = int(round(new_point[0]))
            y_idx = int(round(new_point[1]))
            # Check bounds before marking the waypoint on the mask.
            if 0 <= x_idx < raceline_mask.shape[0] and 0 <= y_idx < raceline_mask.shape[1]:
                raceline_mask[x_idx, y_idx] = True

        # Compute the distance from raceline, for each grid cell
        raceline_cost = distance_transform_edt(~raceline_mask)

        # Final cost map is a sum of the obstacle cost and raceline cost
        # Optionally, we can make this a weighted sum
        cost_map = obstacle_cost + raceline_cost
        cost_map = np.rint(cost_map)
        cost_map = np.clip(cost_map, 0, 100).astype(int)
        
        # print(f"Cost map is shape {cost_map.shape}")
        # print(f"Cost map data is {cost_map.flatten().tolist()}")
        
        self.cost_map.data = cost_map.flatten().tolist() 
        
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
