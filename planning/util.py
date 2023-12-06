from collections import deque
from contest.game import Configuration, Actions

def get_theoretical_legal_successors(position, game_state):
    """
    Returns the legal actions for an agent
    """
    dummy_config = Configuration(position, "North")
    possible_actions = Actions.get_possible_actions(
        dummy_config, game_state.get_walls()
    )

    # Update Configuration
    speed = 1.0

    possible_successors = []
    for action in possible_actions:
        vector = Actions.direction_to_vector(action, speed)
        successor = dummy_config.generate_successor(vector)
        possible_successors.append(successor.pos)

    return possible_successors, possible_actions


# From a list of game states (observation history), extract the first definite position of any enemy agent
def get_first_definite_position(observation_history,agent, current_timeleft, time_limit):
    
    # Iterate through the observation history
    for game_state in reversed(observation_history):
        if game_state.data.timeleft - current_timeleft < time_limit:
            # Iterate through the opponent team
            for opponent in agent.get_opponents(game_state):
                # Get the position of the opponent
                opponent_pos = game_state.get_agent_position(opponent)

                # If the position is not None, return it
                if opponent_pos and game_state.get_agent_state(opponent).is_pacman == False and game_state.get_agent_state(opponent).scared_timer <= 0:
                    return opponent_pos

    # If no definite position was found, return None
    return None

# Get adjacent positions up to a certain distance, by adding and subtracting the distance to the x and y coordinates from (1 to distance)
def get_adjacent_positions(position, distance):
    adjacent_positions = []
    for i in range(1, distance + 1):
        adjacent_positions.append((position[0] + i, position[1]))
        adjacent_positions.append((position[0] - i, position[1]))
        adjacent_positions.append((position[0], position[1] + i))
        adjacent_positions.append((position[0], position[1] - i))
    return adjacent_positions


# Check if position is legal (not a wall) and not negative and not out of bounds based on board width and height
def is_legal_position(position, game_state):
    return (
        position[0] >= 0
        and position[1] >= 0
        and position[0] < game_state.data.layout.width
        and position[1] < game_state.data.layout.height
        and not game_state.has_wall(position[0], position[1])
    )


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