
# Dijkstra Path Planning – Obstacle Course

This project implements **Dijkstra’s algorithm** to compute the shortest path between a start and goal position on a 2D grid map while avoiding obstacles.  
The obstacle environment is constructed using **semi‑algebraic sets** forming the text pattern **“MM 2577”**, with an additional **2 mm collision boundary** around each obstacle.

The algorithm explores the grid, stores visited nodes, reconstructs the optimal path, and animates the search and final path using Matplotlib.

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

python BW-dijkstra_Michael_Mullaney.py

3. The obstacle map will appear and the program will request start and goal coordinates in the terminal.

---

# Input Format

The program expects coordinates in the format:

x,y

Example:

Choose starting x,y coordinates: 10,10  
Choose goal x,y coordinates: 150,40

---

# Map Limits

The workspace is defined as:

0 ≤ x < 180  
0 ≤ y < 50

Coordinates must:

- Be inside the map bounds
- Not lie inside an obstacle
- Not lie inside the 2 mm obstacle boundary

If invalid coordinates are entered, the program will prompt the user again.

---

# Algorithm Overview

The program performs the following steps:

1. Generate the obstacle environment using semi‑algebraic equations.
2. Expand obstacles with a 2 mm collision boundary.
3. Prompt the user for start and goal coordinates.
4. Run **Dijkstra’s algorithm** using a priority queue.
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


