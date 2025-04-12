import csv
import numpy as np
from ament_index_python.packages import get_package_share_directory
from visualization_msgs.msg import Marker, MarkerArray

def load_waypoints(file_path: str) -> np.ndarray:
    '''
    Load waypoints from specified file

    Args:
        file_path (str): The path to the waypoints file
    Returns:
        waypoints (ndarray): The waypoints (x, y, psi, vx)
    '''

    # Get file path
    package_path = get_package_share_directory('f1tenth_mppi')
    file_path = package_path.split("install")[0] + "src/f1tenth_mppi/config/" + file_path

    # Load waypoints
    waypoints = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=';')

        for i in range(3):
            next(csv_reader)

        for i, row in enumerate(csv_reader):
            waypoints.append([float(row[1]), float(row[2]), float(row[3]), 0.5*float(row[5])])

    return np.array(waypoints)

def visualize_waypoints(waypoints, time_stamp, scale=0.2, color=(1.0, 0.0, 0.0)):
    marker_array = MarkerArray()

    for i, (x, y, yaw, v) in enumerate(waypoints):
        marker = Marker()
        marker.header.frame_id = "map"  # or whatever your frame is
        # marker.header.frame_id = "ego_racecar/base_link"
        marker.header.stamp = time_stamp
        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker_array.markers.append(marker)

    return marker_array