# ----------------------------------------------

$ catkin_make

$ source devel/setup.bash
$ roscore

$ source devel/setup.bash
$ roslaunch racecar teleop.launch

$ source devel/setup.bash
$ roslaunch racecar sensors.launch 

$ source devel/setup.bash
$ rosrun vicon_control vision_lanefollower_pid.py

# ----------------------------------------------

$ source devel/setup.bash
$ roslaunch racecar visualization.launch

