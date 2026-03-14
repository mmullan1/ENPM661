# 8-Puzzle Breadth-First Search (BFS) Solver

This program solves the classic **8-puzzle problem** using a **Breadth-First Search (BFS)** algorithm and generates three output files describing the search and solution.

## Overview

The 8-puzzle consists of a 3×3 board containing tiles numbered 1–8 and one blank space (represented as `0`). The objective is to transform a given start configuration into a specified goal configuration by sliding tiles into the blank space.

This implementation performs an exhaustive BFS to guarantee the **shortest solution path (minimum number of moves)**. This method is valid so long as all of the moves
have the same cost to perform

## State Representation

Each puzzle state is stored as a **1×9 vector** (NumPy array):

```
[a b c d e f g h i]
```

When reshaped into a 3×3 board:

```
a b c
d e f
g h i
```

The blank tile is represented by `0`.

## Core Data Structures

### Open List
A FIFO (First Input First Output) queue implemented as a Python list that stores frontier nodes for BFS expansion.

Each entry contains:
```
(state_vector, node_index)
```

### Closed List
Stores visited states (as tuples) to prevent revisiting and infinite loops.

### Parents Dictionary
Maps each discovered state to:
```
(parent_state_tuple, node_index, parent_node_index)
```
This enables reconstruction of the solution path after the goal is found.

## Movement Model

The blank tile can move in four directions if within bounds:

- Up
- Down
- Left
- Right

Each move swaps the blank with an adjacent tile to produce a new state. If a move is instructed that is impossible
(which shouldn't happen), the program will identify that and reject the motion, notifying the user of the error

## BFS Procedure

1. Insert the start state into the open list.
2. Repeatedly:
   - Remove the front state.
   - Check if it matches the goal.
   - Generate valid moves.
   - Add unseen states to the open list.
   - Record parent relationships.
3. Stop when the goal is reached.

Because BFS explores level-by-level, the first solution found is optimal.

## Path Reconstruction

After reaching the goal:

- Follow parent links from goal back to start.
- Reverse the order to obtain the start → goal sequence.
- Convert each state to a flat 1×9 representation.

## Output Files

### Nodes.txt
Contains all explored states (one per line):

```
1 2 3 0 5 6 4 7 8
...
```

### NodesInfo.txt
Contains node metadata:

```
Node Index | Parent Node Index | Node
```

Each row lists:

- Node index
- Parent node index
- Corresponding state (1×9)

### nodePath.txt
Contains the solution path from start to goal:

- One state per line
- Each state written as a flat 1×9 vector

Example:

```
7 5 4 0 3 2 8 1 6
7 5 4 3 0 2 8 1 6
...
1 2 3 4 5 6 7 8 0
```

## Running the Program

Edit the start and goal states in the main block:

```python
start_state = np.array([...])
goal_state  = np.array([...])
```

Then execute:

```
python your_script.py
```

The three output files will be generated in the current directory.

## Libraries Used

The `numpy` library is used to efficiently manipulate puzzle states as arrays. Specifically, it enables reshaping between 1×9 vector form and 3×3 grid form, flattening states for storage and output, and performing fast element-wise comparisons when checking whether a state matches the goal configuration.

The `sys` module is used to terminate the program immediately once the goal state is found and the solution path has been generated, preventing any further unnecessary computation.
