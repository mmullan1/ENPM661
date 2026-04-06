
# A* Planning – Obstacle Course

This project implements **A* Algorithm** to compute the shortest path between a start and goal position on a 2D grid map while avoiding obstacles.  
The obstacle environment is constructed using **semi‑algebraic sets** forming the text pattern **“MM 2577”**, with an additional configurable boundary around each obstacle,
enabling the program to simulate how a circular robot would operate instead of just a point robot.

The algorithm explores the grid, stores visited nodes, reconstructs the optimal path, and animates the search and final path using Matplotlib.


Team Members:
Michael Mullaney
UID: 122332577

# GitHub Repository
GitHub Repository: https://github.com/mmullan1/ENPM661/tree/main/Project3
---

# Dependencies

The following Python libraries are required:

- heapq (standard Python library)
- numpy
- matplotlib
- time (standard Python library)

Install required packages if needed:

pip install numpy matplotlib

---

# How to Run the Code

1. Open a terminal in the directory containing the Python script.

2. Run the program:

python a_star_michael.py

3. The obstacle map will appear and the program will request start and goal coordinates in the terminal.

---

# Input Format

The program expects coordinates in the format:

x,y,d

where x and y are cartesian coordinates, and d is the orientation (deg), which must be a multiple of 30

The program then asks the user to input the clearance around the obstacles (i.e., the robot's radius)

Last, the user will input the step size that each step of the solver takes

Example (this is the one shown in the mp4 video):

Choose starting x,y, d coordinates: 20,20,30
Choose goal x,y,d coordinates: 400,150,0
Choose the clearance around the obstacles (also is the robot's radius):5
Choose step size (1 to 10):10
---

# Map Limits

The workspace is defined as:

0 ≤ x < 600
0 ≤ y < 250

Coordinates must:

- Be inside the map bounds
- Not lie inside an obstacle

If invalid coordinates are entered, the program will prompt the user again.

---

# Algorithm Overview

The program performs the following steps:

1. Generate the obstacle environment using semi‑algebraic equations.
2. Expand obstacles with by the radius of the robot.
3. Prompt the user for start and goal coordinates.
4. Run **A* algorithm** using a priority queue.
5. Record explored nodes and their parent nodes.
6. Reconstruct the optimal path after reaching the goal.
7. Animate the explored nodes and final path.

---

# Output

The program produces:

- A visualization of the obstacle environment
- Animation of the search process
- Final optimal path displayed in red

Console output includes:

GOAL REACHED; COST IS <value>  
The algorithm takes <time> s to run

---


