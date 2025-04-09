#!/usr/bin/python3

import numpy as np

import rclpy
from rclpy.node import Node
from message_filters import Subscriber, TimeSynchronizer

from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from f1tenth_mppi.utils import *
from f1tenth_mppi.dynamics_models import KBM

class MPPI(Node):
    def __init__(self):
        super().__init__('MPPI')

        # TODO: Set/Load Parameters
        
        # TODO: Initialize Dynamics Model
            # use f1tenth_mppi.dynamics_models.KBM

        # TODO: Load Waypoints
            # use f1tenth_mppi.utils.load_waypoints()

        # TODO: Create Publishers
            # AckermannDriveStamped
            # OccupancyGrid
            # MarkerArray

        # TODO: Create message_filters Subscribers
            # Odometry
            # LaserScan

        # TODO: Create time synchronized callback

    def callback(self, scan_msg: LaserScan, pose_msg: Odometry):
        '''
        Time synchronized callback to handle LaserScan and Odometry messages

        Args:
            scan_msg (LaserScan): LaserScan data from the LiDAR
            pose_msg (Odometry): Odometry message from the particle filter
        '''

        # TODO: Update parameters

        # TODO: Create Occupancy Grid

        # TODO: Create Cost Map

        # TODO: Create Trajectories

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

    def publish_markers(self, points: np.ndarray, color : np.ndarray = np.array([1.0, 0.0, 0.0])):
        '''
        Publish MarkerArray message

        Args:
            points (ndarray): The points to publish
            color (ndarray): The color of the points
        '''
        
        # TODO: Publish MarkerArray Message

        return

def main(args=None):
    rclpy.init(args=args)
    rrt_node = MPPI()
    rclpy.spin(rrt_node)

    rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
