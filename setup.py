from setuptools import setup

package_name = 'f1tenth_mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tdetlefs',
    maintainer_email='tdetlefs@andrew.cmu.edu',
    description='An implementation of MPPI for the course 16-663 F1Tenth',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mppi_node = f1tenth_mppi.mppi_node:main',
            'mppi_node_new = f1tenth_mppi.mppi_node_new:main',
            'mppi_node_gpu = f1tenth_mppi.mppi_node_gpu:main',
        ],
    },
)
