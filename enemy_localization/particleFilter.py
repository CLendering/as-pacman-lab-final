import numpy as np
from enemy_localization.customLogging import logging, console_log_handler, DeferredFileHandler, LOGGING_ENABLED
from collections import deque
from contest.capture import SONAR_NOISE_VALUES, SIGHT_RANGE
from contest.game import Directions, Configuration, Actions


_MAX_NOISE = max(SONAR_NOISE_VALUES)

class EnemyPositionParticleFilter:
    def __init__(self, num_particles, noisy_position_distribution_buffer_length, initial_game_state, tracked_enemy_index):
        self.__initial_game_state = initial_game_state
        # Possible directions
        self.possible_directions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1], [0, 0]])
        # Mask which is False for all directions except the stop direction 
        self.stop_direction_index = np.where(np.all(self.possible_directions==[0,0], axis=1))[0][0]
        self.stop_direction_mask = np.full((1, 1, len(self.possible_directions)), False, dtype=bool)
        self.stop_direction_mask[:, :, self.stop_direction_index] = True

        walls = initial_game_state.get_walls()
        initial_position = initial_game_state.get_agent_position(tracked_enemy_index)


        # Distributions based on noisy measurements, not actual particle positions
        # probability distributions of possible positions calculated from noisy distance estimates and agent position
        # this is used as further input to update the belief (particles) of the particle filter when resampling
        self.initial_position_distribution = np.zeros((walls.width, walls.height, len(self.possible_directions)))
        self.initial_position_distribution[initial_position[0], initial_position[1], self.stop_direction_index] = 1
        self.noisy_position_distributions = deque([np.copy(self.initial_position_distribution) for _ in range(noisy_position_distribution_buffer_length)], maxlen=noisy_position_distribution_buffer_length)


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
        # Normalize probabilities where the sum is non-zero (ignoring wall positions where we would divide by 0)
        for x in range(walls.width):
            for y in range(walls.height):
                for prev_dir_index in range(len(self.possible_directions)):
                    sum_probabilities = np.sum(self.direction_probabilities[x, y, prev_dir_index])
                    if sum_probabilities > 0:
                        self.direction_probabilities[x, y, prev_dir_index] /= sum_probabilities




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
            enemy_side_columns = np.array(range(mid_point))
            our_side_columns = np.array(range(mid_point, self.walls.width))
            enemy_side_border_column = mid_point - 1
            our_side_border_column = mid_point
        else:
            # If the enemy's spawn position is in the right half, that's their side
            enemy_side_columns = np.array(range(mid_point, self.walls.width))
            our_side_columns = np.array(range(mid_point))
            enemy_side_border_column = mid_point
            our_side_border_column = mid_point - 1
        
        # Boolean arrays where only the relevant positions are true
        self.our_side_mask = np.full((walls.width, walls.height), False, dtype=bool)
        self.our_side_mask[our_side_columns, :] = True
        self.enemy_side_mask = np.full((walls.width, walls.height), False, dtype=bool)
        self.enemy_side_mask[enemy_side_columns, :] = True
        self.our_border_mask = np.full((walls.width, walls.height), False, dtype=bool)
        self.our_border_mask[our_side_border_column, :] = True
        self.enemy_border_mask = np.full((walls.width, walls.height), False, dtype=bool)
        self.enemy_border_mask[enemy_side_border_column, :] = True

        if LOGGING_ENABLED:
            self.logger = logging.getLogger(f'EPPF (enemy {tracked_enemy_index})')
            self.logger.setLevel(logging.WARNING)
            self.logger.addHandler(console_log_handler)
            self.estimated_positions_logger = logging.getLogger(f'estimated positions enemy {tracked_enemy_index})')
            self.estimated_positions_logger.addHandler(DeferredFileHandler(f'estimated_positions_enemy_{tracked_enemy_index}'))
            self.estimated_positions_logger.setLevel(logging.DEBUG)

            self.true_positions_logger = logging.getLogger(f'true positions enemy {tracked_enemy_index})')
            self.true_positions_logger.addHandler(DeferredFileHandler(f'true_positions_enemy_{tracked_enemy_index}'))
            self.true_positions_logger.setLevel(logging.DEBUG)

    def initializedFor(self, game_state):
        return self.__initial_game_state == game_state
    
    def writeLogFiles(self):
        if LOGGING_ENABLED:
            for handler in [*self.estimated_positions_logger.handlers, *self.true_positions_logger.handlers]:
                if type(handler) is DeferredFileHandler:
                    handler.flush()

    def move_particles(self):
        """
        Move particles within the allowed range and considering the map's walls.
        Should be called exactly once after the enemy has actually moved. (e.g. agent 2 calls this in his turn to update enemy 1's position)
        """  
        new_directions = self.__generate_next_directions(self.particles, self.directions)
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
        np.copyto(self.directions, self.__generate_next_directions(self.particles))
        self.was_pacman = is_pacman

        exact_probability_distribution = np.zeros((self.walls.width, self.walls.height, len(self.possible_directions)))
        exact_probability_distribution[position[0], position[1], self.stop_direction_index] = 1
        # Reset noisy position distributions
        for i in range(len(self.noisy_position_distributions)):
            np.copyto(self.noisy_position_distributions[i], exact_probability_distribution)

    
    def reset_to_spawn(self):
        """
        Call this when an enemy has been killed to reset 
        the particles and position distributions to their initial position.
        """
        self.update_with_exact_position(self.spawn_position, is_pacman=False)

    
    def update_with_noisy_distance(self, agent_position, noisy_distance, enemy_who_just_moved, is_pacman):
        """
        Recalculates weights and resamples particles with the given noisy distance.
        Every agent should call this in every turn with all noisy estimates they get.

        If an agent gets an exact position estimate, 
        they should not call this method but update_with_exact_position instead.
        """
        self.__update_noisy_position_distributions(agent_position, noisy_distance, enemy_who_just_moved, is_pacman)
        condensed_position_distribution = self.__get_condensed_position_distribution()
        self.__weigh_particles(condensed_position_distribution)
        self.__resample_particles(condensed_position_distribution)

    
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

    def __update_noisy_position_distributions(self, agent_position, noisy_distance, enemy_who_just_moved, is_pacman):
        """
        Update the noisy position distributions based on a new noisy distance measurement.
        """
        # make old position distributions fade into adjacent cells
        if enemy_who_just_moved == self.tracked_enemy_index:
            self.__flatten_out_noisy_position_distributions()

        # add new position distribution
        self.__add_new_noisy_position_distribution(agent_position, noisy_distance, is_pacman)

        # reset probabilities of positions in SIGHT_RANGE to zero
        self.__clear_noisy_probability_distributions_in_sight_range(agent_position)

        # use the information whether the enemy is a pacman or a ghost to further constrain the possible locations
        if enemy_who_just_moved == self.tracked_enemy_index:
            self.__constrain_noisy_probability_distributions_based_on_pacman_state(is_pacman)

        # normalize noisy position distributions (they may not sum to 1 due to the above steps - e.g. clearing positions in sight range is done in every update)
        for distribution in self.noisy_position_distributions:
            if distribution.sum() > 0:
                distribution /= distribution.sum()
 

    def __flatten_out_noisy_position_distributions(self):
        """
        Flatten out the noisy position distributions to account for potential movement as time progresses.
        """
        # Iterate over all except first one in queue because it will be overwritten anyways when the new one is added
        for i in range(1, len(self.noisy_position_distributions)):
            distribution = self.noisy_position_distributions[i]
            # Initialize new distribution
            new_distribution = np.zeros_like(distribution)

            # Flatten distribution for vectorized operations
            distribution_flat = distribution.reshape(-1)
            non_zero_entries = np.argwhere(distribution_flat > 0)

            # Unravel indices to get x, y, and direction indices
            x_coords, y_coords, direction_indices = np.unravel_index(non_zero_entries[:, 0], distribution.shape)

            current_probs = distribution_flat[non_zero_entries[:, 0]]

            for new_direction_index, new_direction in enumerate(self.possible_directions):
                # Compute new positions to which the probabilities will move
                # clip is used to prevent positions from going out of bounds when updating new_distribution
                # this isn't a problem because self.direction_probabilities is accessed through position, old directions, new direction; not through clipped new_x/new_y
                new_x = np.clip(x_coords + new_direction[0], 0, self.walls.width - 1)
                new_y = np.clip(y_coords + new_direction[1], 0, self.walls.height - 1)

                # Transition probabilities for the current direction to the new direction
                transition_probs = self.direction_probabilities[x_coords, y_coords, direction_indices, new_direction_index]

                # Calculate the probabilities to add
                probs_to_add = current_probs * transition_probs

                # Update the new distribution
                np.add.at(new_distribution, (new_x, new_y, np.full(new_x.shape, new_direction_index)), probs_to_add)


            # Update the old distribution
            np.copyto(distribution, new_distribution)


    def __add_new_noisy_position_distribution(self, agent_position, noisy_distance, is_pacman):
        new_noisy_position_distribution = np.zeros_like(self.initial_position_distribution)

        # Calculate bounds for noise range
        max_distance = noisy_distance + _MAX_NOISE
        min_distance = noisy_distance - _MAX_NOISE

        # Use the pre-calculated Manhattan distance grid 
        # (contains distances between all pairs of positions, np.nan for distances involving walls)
        distance_from_agent = self.manhattan_distance_grid[agent_position[0], agent_position[1], :, :]
        
        # Identify valid positions within noise range and not walls
        valid_positions_mask = (min_distance <= distance_from_agent) & (distance_from_agent <= max_distance)

        # Further mask valid positions based on whether the enemy is a pacman or a ghost
        if is_pacman:
            valid_positions_mask &= self.our_side_mask
        else:
            valid_positions_mask &= self.enemy_side_mask


        # Calculate mask so only probability cells with direction=stop are set
        # (because we don't know which direction they are headed to. but they will start moving when __flatten_out_noisy_position_distributions updates the distributions in the next step)
        final_mask = np.expand_dims(valid_positions_mask, axis=-1) & self.stop_direction_mask

        # Update noisy_position_distribution
        # Each possible position is equally probable
        new_noisy_position_distribution[final_mask] += 1

        # Normalize
        new_noisy_position_distribution /= new_noisy_position_distribution.sum()

        # Add new noisy position distribution to the buffer
        self.noisy_position_distributions.append(new_noisy_position_distribution)


    def __clear_noisy_probability_distributions_in_sight_range(self, agent_position):
        """
        Update all probability distributions such that positions within SIGHT_RANGE of agent_position are set to zero.
        """
        # Use the pre-calculated Manhattan distance grid
        distances_from_agent_to_all_positions = self.manhattan_distance_grid[agent_position[0], agent_position[1], :, :]

        # Positions within SIGHT_RANGE and not walls
        within_sight_range_mask = distances_from_agent_to_all_positions <= SIGHT_RANGE
                                   
        # Set these positions to zero in noisy_position_distributions
        for distribution in self.noisy_position_distributions:
            distribution[within_sight_range_mask] = 0       



    def __constrain_noisy_probability_distributions_based_on_pacman_state(self, is_pacman):
        """
        After the noisy probability distributions have been flattened out because the enemy moved,
        further constrain the possible locations of the enemy based on its state transition (Pacman or Ghost).
        was_pacman     | new is_pacman      | possible locations
        --------------------------------------------------------
        False          | False              | on their side
        False          | True               | just entered our side (exactly at 1 column)
        True           | False              | just entered their side (exactly at 1 column)
        True           | True               | on our side
        """
        for distribution in self.noisy_position_distributions:
            if self.was_pacman and not is_pacman:
                # Enemy was Pacman and now is a ghost, should be just entering their side
                # Only the border column of their side is possible
                distribution[self.enemy_border_mask, :] = 0
            elif not self.was_pacman and is_pacman:
                # Enemy was a ghost and now is Pacman, should be just entering our side
                # Only the border column of our side is possible
                distribution[self.our_border_mask, :] = 0
            elif not self.was_pacman and not is_pacman:
                # Enemy was a ghost and is still a ghost, should be on their side
                # Only columns of their side are possible
                distribution[self.our_side_mask, :] = 0
            elif self.was_pacman and is_pacman:
                # Enemy was Pacman and is still Pacman, should be on our side
                # Only columns of our side are possible
                distribution[self.enemy_side_mask, :] = 0
        
        # Update the was_pacman state for the next iteration
        self.was_pacman = is_pacman


    def _DEBUG_print_noisy_position_distributions(self):
        l = []
        for i, d in enumerate(self.noisy_position_distributions):
            nonzero = np.sum(d,axis=2).nonzero()
            nonzero_pos = [(a, b) for a, b in zip(*nonzero)] if all(len(arr) > 0 for arr in nonzero) else []
            print(f'{i}: {nonzero_pos}')
            l.append(nonzero_pos)
        return l
    
    def _DEBUG_print_noisy_position_distribution(self, distribution):
            nonzero = np.sum(distribution,axis=2).nonzero()
            nonzero_pos = [(a, b) for a, b in zip(*nonzero)] if all(len(arr) > 0 for arr in nonzero) else []
            print(nonzero_pos)
            return nonzero_pos
        

    def __get_condensed_position_distribution(self):
        """
        Returns a condensed version of all distributions in self.noisy_position_distributions.
        This makes the noise cancel out and the distribution more accurate.
        """
        # Sum across all distributions and directions (the directions are just for flattening out the distributions across time) 
        # This way the noise hopefully cancels out
        condensed_distribution = np.sum(np.array(self.noisy_position_distributions), axis=0).sum(axis=2)

        # normalize
        condensed_distribution /= np.sum(self.noisy_position_distributions)
        assert np.isclose(condensed_distribution.sum(), 1)
        return condensed_distribution

    def __weigh_particles(self, condensed_position_distribution):
        """
        Weigh particles based on the condensed position distribution.
        """
        int_particles = np.rint(self.particles).astype(int)
        self.weights[:] = condensed_position_distribution[int_particles[:,0], int_particles[:,1]]
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        else:
            print('WTF?! This is bad and should not happen')
            self.weights[:] = 1/self.num_particles

    def __resample_particles(self, condensed_position_distribution):
        """
        Resample particles based on their weights.
        Also adds random particles based on condensed_position_distribution.
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
        possible_positions = np.argwhere(condensed_position_distribution != 0)
        # Randomly sample from possible positions
        random_particles = possible_positions[np.random.choice(possible_positions.shape[0], num_random_particles)]
        self.particles[num_resampled_particles:] = random_particles
        self.directions[num_resampled_particles:] = self.__generate_next_directions(random_particles)


    def __generate_next_directions(self, current_positions, current_directions=None):
        current_positions = np.rint(current_positions).astype(int)

        # Handling the case when current_directions is None - use (0,0) direction
        if current_directions is None:
            # Index for direction (0,0) in self.possible_directions
            zero_direction_index = np.where((self.possible_directions == [0, 0]).all(axis=1))[0][0]
            direction_indices = np.full(current_positions.shape[0], zero_direction_index)
        else:
            # Convert directions to indices
            direction_indices = np.argmax(np.all(self.possible_directions[:, np.newaxis] == current_directions, axis=2), axis=0)

        # Get the relevant probabilities
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