# f1tenth-mppi
An implementation of MPPI for the course 16-663 F1Tenth

# To-Dos
- ~~Setup GitHub~~ 
- ~~Setup Initial Node~~
- ~~Finish node infrastructure~~
- Create Occupancy Grid
- Create Cost Map
- ~~Create Trajectories~~
- [README]
- [LAUNCH FILE]
- ~~Parameter File~~
- ~~Debug Statements~~

# Launching

```
ros2 run f1tenth_mppi mppi_node.py --ros-args --params-file src/f1tenth_mppi/config/params.yaml
```

To run without info logs

```
ros2 run f1tenth_mppi mppi_node.py --ros-args --params-file src/f1tenth_mppi/config/params.yaml --log-level warn
```