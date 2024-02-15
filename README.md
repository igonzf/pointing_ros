# pointing_ros

## Installation

```shell
$ cd ~/ros2_ws/src
$ git clone --recurse-submodules https://github.com/igonzf/pointing_ros.git
$ pip3 install -r pointing_ros/yolov8_ros/requirements.txt
$ cd ~/ros2_ws
$ rosdep install --from-paths src --ignore-src -r -y
$ colcon build
```

## Usage 

```shell
$ ros2 launch pointing_bringup pointing.launch.py
```