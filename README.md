# MPPI for F1Tenth
An implementation of Model Predictive Path Integral Control (MPPI) as a local planner for the F1Tenth autonomous racing stack.

## Installation

Navigate to the `src` directory of your workspace and clone this repository.

```
cd ~/ros2_ws/src
git clone git@github.com:t-detlefsen/f1tenth_mppi.git
```

Add the global raceline (in the format of [f1tenth_racetracks](https://github.com/f1tenth/f1tenth_racetracks)) to the [`config`](config) directory and update [`config/params.yaml`](config/params.yaml).

Finally build the package

```
colcon build --packages-select f1tenth_mppi
```

## Use

The file [`config/params.yaml`](config/params.yaml) provides a number of parameters that can be modified to tune the performance of MPPI.

To run MPPI, run the following command
```
ros2 run f1tenth_mppi mppi_node.py --ros-args --params-file src/f1tenth_mppi/config/params.yaml
```

To suppress info logs, the log level can be modified
```
ros2 run f1tenth_mppi mppi_node.py --ros-args --params-file src/f1tenth_mppi/config/params.yaml --log-level warn
```

## Visualization

The [`config/params.yaml`](configuration file) contains an argument called 'visualize' which determines whether to publish visualization markers to RViz.

`map` - A processed occupancy grid overlayed on the vehicle

`trajectory` - The given global raceline (red dotted), sampled trajectories (blue dashed), and calculated trajectory (red dashed)

<img src="https://github.com/t-detlefsen/f1tenth_mppi/blob/main/img/mppi_obs.gif" width="650">

## Acknowledgments
- [MizuhoAOKI](https://github.com/MizuhoAOKI) for their work on [python_simple_mppi](https://github.com/MizuhoAOKI/python_simple_mppi)
- Dr. John Dolan for his advice and organization of the F1Tenth course
- Kai Yun + Jixiang Li for their advice as course TAs