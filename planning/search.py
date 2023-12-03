from collections import deque
from planning.util import get_theoretical_legal_successors
import contest.util as util


def bfs_until_non_wall(start, game_state):
    """
    Perform a breadth-first search until a non-wall position is found.

    :param start: Tuple (x, y) representing the start coordinate.
    :return: List of tuples representing the path to the first non-wall position.
    """

    # Correct start if its coordinates are negative
    if start[0] < 0:
        start = (0, start[1])
    if start[1] < 0:
        start = (start[0], 0)

    # Correct start if its coordinates are greater than the width or height of the board
    if start[0] >= game_state.data.layout.width:
        start = (game_state.data.layout.width - 1, start[1])
    if start[1] >= game_state.data.layout.height:
        start = (start[0], game_state.data.layout.height - 1)
        
    # Define movements: right, left, up, down
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Queue for BFS, initialized with the start position
    queue = deque([start])

    # Dictionary to keep track of visited nodes and their parents
    visited = {start: None}

    # Breadth-First Search
    while queue:
        current = queue.popleft()
        
        # check if current is in a valid position
        if current[1] >= game_state.data.layout.height or current[0] >= game_state.data.layout.width:
            continue

        # Check if current is not negative in any coordinate
        if current[0] < 0 or current[1] < 0:
            continue

        # Stop if the current position is not a wall
        if not game_state.has_wall(current[0], current[1]):
            path = []
            while current:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Return reversed path

        # Check each neighbor
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)

            # If the neighbor is not visited, add it to the queue and mark as visited
            if neighbor not in visited:
                queue.append(neighbor)
                visited[neighbor] = current

    # If a non-wall position is not reachable
    return None

def get_actions(goal_state, parents):
    """
    Retrieves the sequence of actions taken to reach the given goal state.

    Parameters:
    - goal_state: The goal state.
    - parents: A dictionary mapping a state to its parent state and the action taken to reach it.

    Returns:
    - A list of actions leading to the goal state.
    """

    # Initialize the list of actions
    actions_list = []

    # Start with the goal state
    cur_state = goal_state

    # Trace back the actions taken to reach the goal state
    while True:
        # Break the loop if the current state is not in the parents dictionary
        if cur_state not in parents:
            break
        # Otherwise, update the current state and append the action taken to reach it
        cur_state, act = parents[cur_state]
        actions_list.append(act)

    # Reverse the action list to get the correct order from start to goal
    actions_list.reverse()
    return actions_list


# dijkstra's algorithm
def dijk(agent, goal, game_state=None):
    return 0


def aStarSearch(agent, goal, game_state, heuristic=dijk):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    WEIGHT = 1

    # Dictionary to store each state's parent state and the action taken to reach it
    parents = {}

    # Set to keep track of visited states
    vis = set()

    # Dictionary to keep track of the nodes that are currently in the priority queue
    # and their associated costs (actual path cost and estimated total cost)
    in_queue = {}

    # Create a priority queue to manage states based on their estimated total cost
    q = util.PriorityQueue()

    # Get the starting state
    agent_pos = game_state.get_agent_position(agent.index)

    # Calculate the heuristic value for the starting state
    # time_heuristic = time.perf_counter()
    heuristic_val = heuristic(agent, goal, game_state)
    # print("Heuristic time: ", time.perf_counter() - time_heuristic)

    # Push the starting state into the queue with its heuristic value and mark it as visited
    q.push(agent_pos, heuristic_val)
    vis.add(agent_pos)

    in_queue[agent_pos] = (0, heuristic_val)

    # Iterate over the queue until it is empty
    while not q.isEmpty():
        # Pop the state with the least estimated total cost from the priority queue
        cur_pos = q.pop()
        cur_path_cost, _ = in_queue.pop(
            cur_pos
        )  # Retrieve and remove the current path cost from the in_queue dictionary

        # If this state is the goal state, set it as goal_state and break from the loop
        if goal == cur_pos:
            goal_state = cur_pos
            break

        legal_successors, legal_actions = get_theoretical_legal_successors(
            cur_pos, game_state
        )

        successors = []
        for successor, action in zip(legal_successors, legal_actions):
            successors.append((successor, action, 1))

        # Iterate through the successors of the current state
        for pos, action, cost in successors:
            # Calculate the total path cost to reach the successor
            path_cost = cur_path_cost + cost

            # Calculate the estimated total cost for the successor (path cost + heuristic)
            # time_heuristic = time.perf_counter()
            total_cost =  WEIGHT * path_cost + heuristic(agent, goal, game_state)
            # print("Heuristic time: ", time.perf_counter() - time_heuristic)

            # If the successor is already in the priority queue and has a higher estimated total cost
            if pos in in_queue:
                if total_cost < in_queue[pos][1]:
                    # Update the estimated total cost in the priority queue and update the parent and action leading to the successor
                    q.update(pos, total_cost)
                    parents[pos] = (cur_pos, action)
            elif pos not in vis:
                # Mark the successor as visited
                vis.add(pos)

                # Store the successor's path cost and estimated total cost in the in_queue dictionary
                in_queue[pos] = (path_cost, total_cost)

                # Push the successor into the priority queue with its estimated total cost
                q.push(pos, total_cost)

                # Store the current state and action leading to the successor in the parents dictionary
                parents[pos] = (cur_pos, action)

    # Return the sequence of actions leading to the goal state
    return get_actions(goal_state, parents)
