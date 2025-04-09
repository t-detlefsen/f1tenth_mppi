import csv
import numpy as np
from ament_index_python.packages import get_package_share_directory

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