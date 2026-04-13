# Project 4 - Panda Pick and Place

## Description

This ROS2 package implements a pick-and-place style motion sequence for the Franka Emika Panda robot. The system uses ROS2 action clients to command both the arm and gripper through predefined joint configurations.

The robot moves through a sequence of poses, simulating object manipulation by opening and closing the gripper at specific waypoints.

---

## Package Structure

```
ENPM661/
└── Project4/
    ├── package_122332577/
    │   └── package_michael/
    │       ├── resource/
    │       │   └── package_michael  
    │       ├── package.xml
    │       ├── setup.py
    │       ├── setup.cfg
    │       ├── pick_and_place.py
    │       └── __init__.py
    └── README.txt  
```

---

## Features

* Uses FollowJointTrajectory for arm motion
* Uses GripperCommand for gripper control
* Executes a predefined sequence:

  * Move to home position
  * Move to goal position 1
  * Close gripper
  * Move to goal position 2
  * Open gripper
  * Return home

---

## How to Run

### 1. Build the workspace

```bash
colcon build
```

### 2. Source the workspace

```bash
source install/setup.bash
```

### 3. Run the node

```bash
ros2 run package_michael pick_and_place
```

---

## Requirements

* ROS2 Humble
* Panda robot simulation (MoveIt / RViz setup)
* Active controllers:

  * /panda_arm_controller/follow_joint_trajectory
  * /panda_hand_controller/gripper_cmd

---

## Notes

* Joint positions are hardcoded for demonstration purposes
* Timing between actions is handled using simple delays
* When running the code, it is "package_michael" instead of "package_122332577"; that's because I didn't see that instruction for the naming until later

---

## Demonstration

See the accompanying PDF for:

* Video constructing the scene from the MoveIt wizard
* Video demonstrations of the system in action
* GitHub Link
