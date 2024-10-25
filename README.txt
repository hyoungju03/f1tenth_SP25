# ----------------------------------------------

$ catkin_make

$ source devel/setup.bash
$ roscore

$ source devel/setup.bash
$ python3 vicon_bridge.py

$ source devel/setup.bash
$ roslaunch racecar teleop.launch

$ source devel/setup.bash
$ roslaunch racecar sensors.launch 

$ source devel/setup.bash
$ rosrun vicon_control vicon_tracker_pp.py

# ----------------------------------------------

$ source devel/setup.bash
$ roslaunch racecar visualization.launch

