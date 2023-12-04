import numpy as np
from enemy_localization.customLogging import logging, console_log_handler, DeferredFileHandler
from collections import deque
from contest.capture import SONAR_NOISE_VALUES, SIGHT_RANGE
from contest.pacman import GhostRules
from contest.game import Directions, Configuration, Actions
from contest.util import Counter

_LOGGING = False

_MAX_NOISE = max(SONAR_NOISE_VALUES)

class EnemyPositionParticleFilter:
    def __init__(self, num_particles, walls, initial_position, tracked_enemy_index):
        # Unnormalized probability distribution based on noisy measurements, 
        # clearing out positions within SIGHT_RANGE of the agent's position,
        # and constraining the possible locations based on the transitions of the enemy's state (Pacman or Ghost)
        # This is used to weigh the particles when resampling
        self.noisy_position_distribution = np.zeros((walls.width, walls.height))
        self.noisy_position_distribution[initial_position] = 1

        # dtype needs to be float because scared ghosts move slower (0.5), normal speed is 1.0 
        self.particles = np.full((num_particles, 2), initial_position, dtype=float)
        self.weights = np.full(num_particles, 1/num_particles)
        self.num_particles = num_particles

        self.walls = walls
        self.spawn_position = initial_position
        self.tracked_enemy_index = tracked_enemy_index

        # Precompute manhattan distance grid for faster distance calculations (contains distances between all pairs of positions)
        self.manhattan_distance_grid = np.empty((walls.width, walls.height, walls.width, walls.height))
        for i in range(walls.width):
            for j in range(walls.height):
                for k in range(walls.width):
                    for l in range(walls.height):
                        self.manhattan_distance_grid[i, j, k, l] = abs(i - k) + abs(j - l)
        # Set distances involving walls to nan
        wall_positions = np.where(walls.data)
        for x, y in zip(*wall_positions):
            self.manhattan_distance_grid[x, y, :, :] = np.nan
            self.manhattan_distance_grid[:, :, x, y] = np.nan


        # Possible directions
        self.possible_directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]])
        # Constants for probabilities of how directions transition from one time step to the next
        TURN_OR_MOVEMENT_START_PROBABILITY = 0.3
        REVERSE_PROBABILITY = 0.1
        STOP_PROBABILITY = 0.05
        # Precompute direction probabilities for each non-wall grid position and previous direction
        self.direction_probabilities = np.zeros((walls.width, walls.height, len(self.possible_directions), len(self.possible_directions)))
        # Get non-wall positions
        non_wall_positions = np.argwhere(~np.array(walls.data))
        for x, y in non_wall_positions:
            for prev_dir_index, prev_direction in enumerate(self.possible_directions):
                for new_dir_index, new_direction in enumerate(self.possible_directions):
                    # Check if the new direction is possible (not leading into a wall)
                    new_x, new_y = x + new_direction[0], y + new_direction[1]
                    if 0 <= new_x < walls.width and 0 <= new_y < walls.height and not walls[new_x][new_y]:
                        # Assign probabilities based on direction changes
                        # From least to most probable:
                        # - stopping
                        # - reversing
                        # - turning or starting movement
                        # - continuing movement in the same direction
                        if np.array_equal(new_direction, [0, 0]):
                            self.direction_probabilities[x, y, prev_dir_index, new_dir_index] = STOP_PROBABILITY
                        elif _is_reverse(new_direction, prev_direction):
                            self.direction_probabilities[x, y, prev_dir_index, new_dir_index] = REVERSE_PROBABILITY
                        elif _is_turn_or_movement_start(new_direction, prev_direction):
                            self.direction_probabilities[x, y, prev_dir_index, new_dir_index] = TURN_OR_MOVEMENT_START_PROBABILITY
                        else:
                            remaining_prob = 1 - TURN_OR_MOVEMENT_START_PROBABILITY - REVERSE_PROBABILITY - STOP_PROBABILITY
                            self.direction_probabilities[x, y, prev_dir_index, new_dir_index] = remaining_prob
                    else:
                        # Set probability to zero for impossible directions
                        self.direction_probabilities[x, y, prev_dir_index, new_dir_index] = 0
        # Normalize probabilities
        self.direction_probabilities /= np.sum(self.direction_probabilities, axis=-1, keepdims=True)


        # Initialize particle directions
        dummy_config = Configuration(initial_position, Directions.STOP)
        legal_directions = Actions.get_possible_actions(dummy_config, walls)
        # strict bias towards non-zero directions
        legal_directions = np.array([Actions.direction_to_vector(action) for action in legal_directions if action != Directions.STOP])
        # Randomly select non-zero directions for particles
        indices = np.random.choice(len(legal_directions), size=num_particles)
        self.directions =  legal_directions[indices]

        # Whether the enemy was observed to be a pacman in the last update
        self.was_pacman = False
        self.scared_timer = 0 # TODO update scared_timer and use it to change the speed of particles


        # Determine the columns that belong to our side and the enemy's side based on the spawn position.
        mid_point = self.walls.width // 2
        if self.spawn_position[0] < mid_point:
            # If the enemy's spawn position is in the left half, that's their side
            self.enemy_side_columns = np.array(range(mid_point))
            self.our_side_columns = np.array(range(mid_point, self.walls.width))
            enemy_side_border_column = mid_point - 1
            our_side_border_column = mid_point
        else:
            # If the enemy's spawn position is in the right half, that's their side
            self.enemy_side_columns = np.array(range(mid_point, self.walls.width))
            self.our_side_columns = np.array(range(mid_point))
            enemy_side_border_column = mid_point
            our_side_border_column = mid_point - 1
        self.all_columns_except_border_of_our_team = np.setdiff1d(np.arange(self.walls.width), our_side_border_column)
        self.all_columns_except_border_of_enemy_team = np.setdiff1d(np.arange(self.walls.width), enemy_side_border_column)


        if _LOGGING:
            self.logger = logging.getLogger(f'EPPF (enemy {tracked_enemy_index})')
            self.logger.setLevel(logging.WARNING)
            self.logger.addHandler(console_log_handler)
            self.estimated_positions_logger = logging.getLogger(f'estimated positions enemy {tracked_enemy_index})')
            self.estimated_positions_logger.addHandler(DeferredFileHandler(f'estimated_positions_enemy_{tracked_enemy_index}'))
            self.estimated_positions_logger.setLevel(logging.DEBUG)

            self.true_positions_logger = logging.getLogger(f'true positions enemy {tracked_enemy_index})')
            self.true_positions_logger.addHandler(DeferredFileHandler(f'true_positions_enemy_{tracked_enemy_index}'))
            self.true_positions_logger.setLevel(logging.DEBUG)

    
    def writeLogFiles(self):
        if _LOGGING:
            for handler in [*self.estimated_positions_logger.handlers, *self.true_positions_logger.handlers]:
                if type(handler) is DeferredFileHandler:
                    handler.flush()

    def move_particles(self):
        """
        Move particles within the allowed range and considering the map's walls.
        Should be called exactly once after the enemy has actually moved. (e.g. agent 2 calls this in his turn to update enemy 1's position)
        """  
        new_directions = self.__generate_random_directions(self.particles, self.directions)
        np.copyto(self.directions, new_directions)
        self.particles += new_directions # TODO multiply with speed depending on scared timer and is_pacman
        
    
    def update_with_exact_position(self, position, is_pacman):
        """
        Sets all particles to an exactly known position.
        Call this when:
        - An enemy agent is directly seen and we get their exact position.
        - An enemy agent has been killed. They'll respawn at their spawn point,
          so we can set all particles to that position (saved in EnemyPositionParticleFilter.spawn_position)
        """
        self.particles[:] = position
        self.weights[:] = 1/self.num_particles
        np.copyto(self.directions, self.__generate_random_directions(self.particles))
        self.was_pacman = is_pacman

    
    def reset_to_spawn(self):
        """
        Call this when an enemy has been killed to reset 
        the particles and position distributions to their initial position.
        """
        self.update_with_exact_position(self.spawn_position, is_pacman=False)

    
    def update_with_noisy_distance(self, agent_position, noisy_distance, is_pacman):
        """
        Recalculates weights and resamples particles with the given noisy distance.
        Every agent should call this in every turn with all noisy estimates they get.

        If an agent gets an exact position estimate, 
        they should not call this method but update_with_exact_position instead.
        """
        self.__update_noisy_position_distribution(agent_position, noisy_distance, is_pacman)
        self.__weigh_particles()
        self.__resample_particles()

    
    def estimate_distinct_position(self):
        """
        Estimate the position as the mean of the particle distribution.
        Returns a single position.
        """
        mean_position = np.rint(np.mean(self.particles, axis=0)).astype(int)
        # Check if mean position is valid
        if self.__is_valid(mean_position):
            return tuple(mean_position)
        
        # Find the average of the nearest valid particles to the mean position
        distances = np.linalg.norm(self.particles - mean_position, ord=1, axis=1)
        min_distance = np.min(distances)
        nearest_particles = self.particles[distances == min_distance]
        mean_position = np.rint(np.mean(nearest_particles, axis=0)).astype(int)

        # Check if mean position is valid now
        if self.__is_valid(mean_position):
            return tuple(mean_position)
        
        # mean position is still not valid - just chose a random particle for now.
        # TODO save last estimates and in this case pick the particle closest to last estimates
        nearest_position = np.rint(nearest_particles[np.random.choice(len(nearest_particles),1)]).astype(int)[0]
        return tuple(nearest_position)

    def estimate_probabilistic_position(self):
        """
        Generate a probabilistic position estimate based on the particle filter data.
        Returns a array of positions where the value at each position is the probability of that position.
        """
        probability_distribution = np.zeros((self.walls.width, self.walls.height))

        unique_positions, counts = np.unique(np.rint(self.particles).astype(int), axis=0, return_counts=True)

        for i, position in enumerate(unique_positions):
            probability_distribution[tuple(position)] = counts[i] / self.num_particles

        return probability_distribution

    def __update_noisy_position_distribution(self, agent_position, noisy_distance, is_pacman):
        """
        Update the noisy position distributions based on a new noisy distance measurement.
        """
        # set new position distribution from new noisy_distance
        self.__set_new_noisy_position_distribution(agent_position, noisy_distance)

        # reset probabilities of positions in SIGHT_RANGE to zero
        self.__clear_noisy_probability_distribution_in_sight_range(agent_position)

        # use the information whether the enemy is a pacman or a ghost to further constrain the possible locations
        self.__constrain_noisy_probability_distribution_based_on_pacman_state(is_pacman)
 

    def __set_new_noisy_position_distribution(self, agent_position, noisy_distance):
        # TODO check if it's better if we don't reset the distribution to zero but instead just increment the values
        # but I don't think that would be better
        self.noisy_position_distribution[:] = 0

        # Calculate bounds for noise range
        min_distance = noisy_distance - _MAX_NOISE
        max_distance = noisy_distance + _MAX_NOISE

        # Use the pre-calculated Manhattan distance grid 
        # (contains distances between all pairs of positions, np.nan for distances involving walls)
        distance_from_agent = self.manhattan_distance_grid[agent_position[0], agent_position[1], :, :]
        
        # Identify valid positions within noise range and not walls
        valid_positions_mask = distance_from_agent <= max_distance

        # Update noisy_position_distribution
        self.noisy_position_distribution[valid_positions_mask] += 1


    def __clear_noisy_probability_distribution_in_sight_range(self, agent_position):
        """
        Update all probability distributions such that positions within SIGHT_RANGE of agent_position are set to zero.
        """
        # Use the pre-calculated Manhattan distance grid
        distances_from_agent_to_all_positions = self.manhattan_distance_grid[agent_position[0], agent_position[1], :, :]

        # Positions within SIGHT_RANGE and not walls
        within_sight_range_mask = distances_from_agent_to_all_positions <= SIGHT_RANGE
                                   
        # Set these positions to zero in noisy_position_distribution
        self.noisy_position_distribution[within_sight_range_mask] = 0       



    def __constrain_noisy_probability_distribution_based_on_pacman_state(self, is_pacman):
        """
        Constrain the possible locations of the enemy based on its state transition (Pacman or Ghost).
        was_pacman     | new is_pacman      | possible locations
        --------------------------------------------------------
        False          | False              | on their side
        False          | True               | just entered our side (exactly at 1 column)
        True           | False              | just entered their side (exactly at 1 column)
        True           | True               | on our side
        """
        if self.was_pacman and not is_pacman:
            # Enemy was Pacman and now is a ghost, should be just entering their side
            # Only the border column of their side is possible
            self.noisy_position_distribution[self.all_columns_except_border_of_enemy_team, :] = 0
        elif not self.was_pacman and is_pacman:
            # Enemy was a ghost and now is Pacman, should be just entering our side
            # Only the border column of our side is possible
            self.noisy_position_distribution[self.all_columns_except_border_of_our_team, :] = 0
        elif not self.was_pacman and not is_pacman:
            # Enemy was a ghost and is still a ghost, should be on their side
            # Only columns of their side are possible
            self.noisy_position_distribution[self.our_side_columns, :] = 0
        elif self.was_pacman and is_pacman:
            # Enemy was Pacman and is still Pacman, should be on our side
            # Only columns of our side are possible
            self.noisy_position_distribution[self.enemy_side_columns, :] = 0
    
        # Update the was_pacman state for the next iteration
        self.was_pacman = is_pacman
        
    def __weigh_particles(self):
        """
        Weigh particles based on the noisy position distribution.
        """
        int_particles = np.rint(self.particles).astype(int)
        self.weights[:] = self.noisy_position_distribution[int_particles[:,0], int_particles[:,1]]
        self.weights /= self.weights.sum()

    def __resample_particles(self):
        """
        Resample particles based on their weights.
        Also adds random particles based on self.noisy_position_distribution.
        """
        N = self.num_particles
        # TODO tune random particles fraction
        RANDOM_PARTICLES_FRACTION=0.05 
        num_random_particles = int(N * RANDOM_PARTICLES_FRACTION)
        num_resampled_particles = N - num_random_particles
 
        # Resample based on weights for the majority of particles
        indices = _systematic_resample(self.weights, num_resampled_particles)
        self.particles[:num_resampled_particles] = self.particles[indices]
        self.directions[:num_resampled_particles] = self.directions[indices]
 
        # Add random particles from the current noisy position distribution to prevent particle deprivation
        # Find positions with non-zero probability
        possible_positions = np.argwhere(self.noisy_position_distribution != 0)
        # Randomly sample from possible positions
        random_particles = possible_positions[np.random.choice(possible_positions.shape[0], num_random_particles)]
        self.particles[num_resampled_particles:] = random_particles
        self.directions[num_resampled_particles:] = self.__generate_random_directions(random_particles)


    def __generate_random_directions(self, current_positions, current_directions=None):
        # Normalize and ensure positions are within bounds
        current_positions = np.rint(current_positions).astype(int)
        current_positions[:, 0] = np.clip(current_positions[:, 0], 0, self.walls.width - 1)
        current_positions[:, 1] = np.clip(current_positions[:, 1], 0, self.walls.height - 1)

        # Handling the case when current_directions is None - use (0,0) direction
        if current_directions is None:
            # Index for direction (0,0) in self.possible_directions
            zero_direction_index = np.where((self.possible_directions == [0, 0]).all(axis=1))[0][0]
            direction_indices = np.full(current_positions.shape[0], zero_direction_index)
        else:
            # Convert directions to indices
            direction_indices = np.argmax(np.all(self.possible_directions[:, np.newaxis] == current_directions, axis=2), axis=0)

        # Use advanced indexing to get the probabilities
        all_direction_probabilities = self.direction_probabilities[current_positions[:, 0], current_positions[:, 1], direction_indices]

        # Vectorized selection of new direction indices
        new_dir_indices = np.array([np.random.choice(len(self.possible_directions), p=probs) for probs in all_direction_probabilities])

        # Map the indices to actual directions
        new_directions = self.possible_directions[new_dir_indices]

        return new_directions

    def __is_valid(self, position):
        """
        Checks if position is valid for an agent (i.e. it is not inside a wall).
        """
        x, y = np.rint(position).astype(int)
        return not self.walls[x][y]
    


def _is_turn_or_movement_start(new_direction, prev_direction):
    """
    Check if the change from prev_direction to new_direction represents a 90-degree turn,
    accounting for possible changes in speed due to the scared flag of an agent changing.

    Also returns True when prev_direction was not moving and new_direction is moving.
    """
    # A 90-degree turn occurs if the direction changes in one axis but remains constant in the other
    return (prev_direction[0] == new_direction[0] and prev_direction[1] != new_direction[1]) or \
           (prev_direction[1] == new_direction[1] and prev_direction[0] != new_direction[0])

def _is_reverse(new_direction, prev_direction):
    """
    Check if the change from prev_direction to new_direction represents a reversal of direction,
    accounting for possible changes in speed due to the scared flag of an agent changing.
    """
    # A reversal occurs if the new direction is the opposite of the previous one
    return np.array_equal(new_direction, -prev_direction)

def _systematic_resample(weights, n=None):
    N = len(weights)
    n = N if n is None else n
    positions = (np.arange(N) + np.random.random()) / N

    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1 # prevent numerical errors

    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
            if j >= N:  # Prevent index out of bounds
                j = N - 1

    if n < N:
        return np.random.choice(indices, size=n, replace=False)
    
    return indices