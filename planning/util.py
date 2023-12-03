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


