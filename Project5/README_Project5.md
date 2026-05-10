# Project 5: RRT-Based Motion Planning for the Dobot CR3

This project implements sampling-based joint-space motion planning for a simulated Dobot CR3 robot. The planner is written in Python and can interface with a ROS 2 / MoveIt / RViz simulation environment. Several RRT-based algorithms are included, including standard RRT, RRT*, Bi-RRT, Bi-RRT*, lazy variants, and an improved Bi-RRT* variant.

The project supports:

- Loading obstacle scenes from YAML files
- Running a single planning trial
- Saving the generated joint-space path to a CSV file
- Optionally executing the path through the CR3 trajectory controller
- Running bulk benchmark tests over multiple algorithms and environments

---

## 1. Clone the Repository

Clone the repository into your local machine:

```bash
git clone https://github.com/mmullan1/ENPM661.git
```

Move into the Project 5 workspace:

```bash
cd ENPM661/Project5/dobot_ws
```

The main Python files for this project are located in:

```bash
cd project5_scene
```

---

## 2. Required Environment

This project assumes that the user already has a working ROS 2 / MoveIt simulation setup for the Dobot CR3.

The Python planning code uses packages including:

```bash
numpy
matplotlib
pyyaml
scipy
trimesh
rclpy
trajectory_msgs
moveit_msgs
shape_msgs
geometry_msgs
```

If the non-ROS Python dependencies are missing, install them with:

```bash
pip3 install numpy matplotlib pyyaml scipy trimesh
```

The ROS message packages must be installed through the ROS 2 environment.

---

## 3. Source ROS 2

In every new terminal used for ROS commands, source ROS 2:

```bash
source /opt/ros/humble/setup.bash
```

If using a local ROS workspace for the Dobot CR3 simulation, also source the local install space:

```bash
source ~/dobot_ws/install/setup.bash
```

If the repository was cloned somewhere else, update the path accordingly.

---

## 4. Launch the CR3 MoveIt / RViz Simulation

In a separate terminal, launch the Dobot CR3 MoveIt/RViz simulation.

Use the launch command for your CR3 MoveIt configuration. For example:

```bash
ros2 launch cr3_moveit demo.launch.py

```

The important requirement is that MoveIt/RViz is running and listening to the planning scene topic:

```bash
/planning_scene
```

For trajectory execution, the controller should also be listening on:

```bash
/cr3_group_controller/joint_trajectory
```

---

## 5. Load a Scene into MoveIt

From the project folder:

```bash
cd ~/ENPM661/Project5/dobot_ws/project5_scene
```

Load the easy scene:

```bash
python3 load_scene.py lab_scene_easy.yaml
```

Load the harder scene:

```bash
python3 load_scene.py lab_scene.yaml
```

Load the empty scene:

```bash
python3 load_scene.py lab_scene_empty.yaml
```

The scene loader removes previous known scene objects and then publishes the selected YAML scene to MoveIt.

---

## 6. Run a Single Planning Trial

To run one trial, save the generated path, and visualize the result in Matplotlib, use:

```bash
python3 rrt_builder.py --scene lab_scene_easy.yaml --alg rrt --output 2 --motion false
```

This example runs the RRT algorithm in the easy lab scene without executing the motion in the simulation.

After the planner finishes, the resulting path is saved to:

```bash
rrt_path.csv
```

### Scene Options

The `--scene` argument selects which environment file to use:

```bash
--scene lab_scene_easy.yaml
--scene lab_scene.yaml
--scene lab_scene_empty.yaml
```

### Algorithm Options

The `--alg` argument selects which planning algorithm to run:

```bash
--alg preview
--alg rrt
--alg rrt_rigorious
--alg rrt_star
--alg rrt_star_lazy
--alg birrt
--alg birrt_star
--alg birrt_star_lazy
--alg improved_birrt_star
--alg improved_birrt_star_lazy
```

### Output Options

The `--output` argument controls what is returned and plotted:

```bash
--output 0   # show all explored nodes
--output 1   # show the initial found path
--output 2   # show the smoothed path
```

The recommended setting is:

```bash
--output 2
```

### Motion Execution Options

The `--motion` argument controls whether the path is only generated or also executed in the simulation:

```bash
--motion false   # run the planner without executing the robot motion
--motion true    # run the planner and execute the motion in simulation
```


## 7. Run Bulk Benchmark Tests

Bulk benchmark mode runs multiple trials and saves raw and summarized results.

To run 2 trials per algorithm/environment:

```bash
python3 rrt_builder.py --benchmark true --benchmark_runs 2 --step 5.0
```

To run 20 trials per algorithm/environment:

```bash
python3 rrt_builder.py --benchmark true --benchmark_runs 20 --step 5.0
```

To run the full default benchmark count:

```bash
python3 rrt_builder.py --benchmark true --benchmark_runs 200 --step 5.0
```

Benchmark output files are saved as:

```bash
rrt_benchmark_results.csv
rrt_benchmark_summary.csv
rrt_benchmark_summary.png
```

The raw CSV contains every individual run. The summary CSV contains mean cost, standard deviation, time statistics, success rate, and path node statistics.

---

## 9. Example Workflow

A typical full workflow is:

Terminal 1:

```bash
cd ENPM661/Project5/dobot_ws/project5_scene
source /opt/ros/humble/setup.bash
source ~/dobot_ws/install/setup.bash
ros2 launch cr3_moveit demo.launch.py
```

Terminal 2:

```bash
source /opt/ros/humble/setup.bash
cd ~/ENPM661/Project5/dobot_ws/project5_scene
python3 load_scene.py lab_scene_easy.yaml
```

Terminal 3:

```bash
source /opt/ros/humble/setup.bash
cd ~/ENPM661/Project5/dobot_ws/project5_scene
python3 rrt_builder.py --scene lab_scene_easy.yaml --alg birrt_star_lazy --output 2 --motion true --waypoint_time 5.0
```

For benchmarking instead of motion execution:

```bash
python3 rrt_builder.py --benchmark true --benchmark_runs 2 --step 5.0
```

---

## 10. Notes

- Joint configurations are planned in joint space.
- The saved path is written as joint angles in degrees.
- `run_rrt.py` converts the CSV joint angles from degrees to radians before publishing them to the ROS 2 trajectory controller.
- The planner checks collisions using a simplified cylindrical approximation of the Dobot CR3 links.
- Scene obstacles are loaded from YAML files as boxes and/or meshes.`.

---

## 13. Possible Argument Parser Fix

If `argparse` rejects one of the valid algorithm names, check the `choices` list for the `--alg` argument in `rrt_builder.py`.

It should look like this:

```python
choices=[
    "preview",
    "rrt",
    "rrt_rigorious",
    "rrt_star",
    "rrt_star_lazy",
    "birrt",
    "birrt_star",
    "birrt_star_lazy",
    "improved_birrt_star",
    "improved_birrt_star_lazy",
]
```
