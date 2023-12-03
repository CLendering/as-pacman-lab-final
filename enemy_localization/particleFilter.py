import numpy as np
from enemy_localization.customLogging import logging, console_log_handler, DeferredFileHandler
from collections import deque
from contest.capture import SONAR_NOISE_VALUES, SIGHT_RANGE
from contest.pacman import GhostRules
from contest.game import Directions, Configuration, Actions
from contest.util import Counter

_LOGGING = False

# TODO check runtime of particle filter, add way to scale down or completely 
# shut down if it takes too much time and we risk defaulting game bc of that (1 sec max per turn I think)
class EnemyPositionParticleFilter:
    def __init__(self, num_particles, noisy_distances_buffer_length, walls, initial_position, tracked_enemy_index):
        # Distributions based on noisy measurements, not actual particle positions
        # probability distributions of possible positions calculated from noisy distance estimates and agent position
        # this is used as further input to update the belief (particles) of the particle filter
        initial_position_distribution = np.zeros((walls.width, walls.height))
        initial_position_distribution[initial_position] = 1
        self.noisy_position_distributions = deque([np.copy(initial_position_distribution) for _ in range(noisy_distances_buffer_length)], maxlen=noisy_distances_buffer_length)

        # dtype needs to be float because scared ghosts move slower (0.5), normal speed is 1.0 
        self.particles = np.full((num_particles, 2), initial_position, dtype=float)
        self.weights = np.full(num_particles, 1/num_particles)
        self.num_particles = num_particles

        self.walls = walls
        self.spawn_position = initial_position
        self.tracked_enemy_index = tracked_enemy_index

        self.legal_positions = np.array([[x, y] for x in range(self.walls.width) for y in range(self.walls.height) if not self.walls[x][y]])

        # Initialize particle velocities
        dummy_config = Configuration(initial_position, Directions.STOP)
        legal_directions = Actions.get_possible_actions(dummy_config, walls)
        # strict bias towards non-zero velocities
        legal_velocities = np.array([Actions.direction_to_vector(action, GhostRules.GHOST_SPEED) for action in legal_directions if action != Directions.STOP])
        # Handle purely theoretical edge case where no legal moves result in velocities of zero
        if legal_velocities.size == 0:
            self.velocities = np.zeros((num_particles, 2))
        else:
            # Randomly select non-zero velocities for particles
            indices = np.random.choice(len(legal_velocities), size=num_particles)
            self.velocities =  legal_velocities[indices]

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
        if _LOGGING:
            self.logger.info(f'Moving particles')
        current_particles_counter = Counter()
        for pos in self.particles:
            current_particles_counter[tuple(pos)] += 1
        
        sorted_particle_positions = sorted(current_particles_counter.items(), key=lambda x: x[1], reverse=True)
        if _LOGGING:
            self.logger.info(f'Current top 10 particle positions: {dict(sorted_particle_positions[:10])}')

        new_velocities = self.__generate_random_velocities(self.particles, self.velocities)
        np.copyto(self.velocities, new_velocities)
        self.particles += new_velocities
        
        new_particles_counter = Counter()
        for pos in self.particles:
            new_particles_counter[tuple(pos)] += 1

        new_sorted_particle_positions = sorted(new_particles_counter.items(), key=lambda x: x[1], reverse=True)
        if _LOGGING:
            self.logger.info(f'New particle positions: {dict(new_sorted_particle_positions[:10])}')
    
    
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
        np.copyto(self.velocities, self.__generate_random_velocities(self.particles))

        position_distribution = np.zeros((self.walls.width, self.walls.height))
        position_distribution[position] = 1
        for i in range(len(self.noisy_position_distributions)):
            self.noisy_position_distributions[i] = np.copy(position_distribution)
        
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
        self.__update_noisy_position_distributions(agent_position, noisy_distance, is_pacman)
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
        if self.walls[nearest_position[0]][nearest_position[1]]:
            raise "WTF"
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

    def __update_noisy_position_distributions(self, agent_position, noisy_distance, is_pacman):
        """
        Update the noisy position distributions based on a new noisy distance measurement.
        """
        # make old position distributions fade into adjacent cells
        self.__flatten_noisy_position_distributions()

        # add new position distribution
        self.__add_new_noisy_position_distribution(agent_position, noisy_distance)

        # reset probabilities of positions in SIGHT_RANGE to zero
        self.__clear_probabilities_in_sight_range(agent_position)

        # use the information whether the enemy is a pacman or a ghost to further constrain the possible locations
        self.__constrain_based_on_pacman_state(is_pacman)
 
    def __flatten_noisy_position_distributions(self):
        """
        Flatten out the noisy position distributions to account for potential movement as time progresses.
        """
        # Each agent (enemy or friendly) moves once every four turns. They can move at most 1 grid cell per turn. The particle filters for every enemy is updated every two turns. So every time a particle filter is updated, the enemy moves on average by +-0.5  (because it can move by 1 every four turns and the particle filters are updated every two turns). The same goes for the measuring agent. So the true distance of a measuring agent to every enemy can change by |+-0.5|+|+-0.5|=1 every two turns. The position of an enemy agent can change by one of the vectors in [(-0.5, 0), (0, -0.5), (0, 0), (0.5, 0), (0,0.5)] every two turns. 
        # Remember that The position of an enemy agent can change by one of the vectors in [(-0.5, 0), (0, -0.5), (0, 0), (0.5, 0), (0,0.5)] every two turns when the particle filters are being updated.

        # TODO probabilities might oscillate back and forth
        # to mitigate this I'd have to add some temporal/directional memory 

        STOP_PROBABILITY = 0.05
        for i in range(len(self.noisy_position_distributions)):
            distribution = self.noisy_position_distributions[i]
            flattened_distribution = np.zeros_like(distribution)

            for (x, y), probability in np.ndenumerate(distribution):
                # Skip cells which don't have any probability to flatten out in the first place
                if probability == 0:  
                    continue
                
                # Probability of staying in the same cell
                flattened_distribution[x][y] += probability * STOP_PROBABILITY

                # Spread out remaining probability to adjacent cells
                adjacent_cells = self.__get_adjacent_cells(x, y)
                if adjacent_cells:  # Ensure there are adjacent cells
                    move_probability = (probability * (1 - STOP_PROBABILITY)) / len(adjacent_cells)
                    for new_x, new_y in adjacent_cells:
                        flattened_distribution[new_x][new_y] += move_probability

            self.noisy_position_distributions[i] = flattened_distribution

    def __add_new_noisy_position_distribution(self, agent_position, noisy_distance):
        noise_range=max(SONAR_NOISE_VALUES)
        position_distribution = np.zeros((self.walls.width, self.walls.height))

        for x in range(self.walls.width):
            for y in range(self.walls.height):
                if not self.walls[x][y]:
                    distance = abs(x - agent_position[0]) + abs(y - agent_position[1])  # Manhattan distance
                    if noisy_distance - noise_range <= distance <= noisy_distance + noise_range:
                        # Uniform weighting for positions within the noise range
                        position_distribution[x, y] += 1

        # Normalize the distribution
        position_distribution /= position_distribution.sum()

        self.noisy_position_distributions.append(position_distribution)

    def __clear_probabilities_in_sight_range(self, agent_position):
        """
        Update all probability distributions such that positions within SIGHT_RANGE of agent_position are set to zero.
        """


        for distribution in self.noisy_position_distributions:
            changed = []
            old = []
            for x in range(self.walls.width):
                for y in range(self.walls.height):
                    if self.__within_sight_range(agent_position, (x, y)):
                        old.append(distribution[x,y])
                        changed.append((x,y))
                        distribution[x, y] = 0
            # normalize 
            if distribution.sum() > 0:
                distribution /= distribution.sum()
            else:
                print('oo')# TODO remove

    def __within_sight_range(self, agent_pos, pos):
        """
        Check if pos is within SIGHT_RANGE of agent_pos.
        """
        return np.linalg.norm(np.array(agent_pos) - np.array(pos), ord=1) <= SIGHT_RANGE

    def __constrain_based_on_pacman_state(self, is_pacman):
        """
        Constrain the possible locations of the enemy based on its state transition (Pacman or Ghost).
        was_pacman     | new is_pacman      | possible locations
        --------------------------------------------------------
        False          | False              | on their side
        False          | True               | just entered our side (exactly at 1 column)
        True           | False              | just entered their side (exactly at 1 column)
        True           | True               | on our side
        """

        for distribution in self.noisy_position_distributions:
            old_distribution = np.copy(distribution)
            if self.was_pacman and not is_pacman:
                # Enemy was Pacman and now is a ghost, should be just entering their side
                # Only the border column of their side is possible
                distribution[self.all_columns_except_border_of_enemy_team, :] = 0
            elif not self.was_pacman and is_pacman:
                # Enemy was a ghost and now is Pacman, should be just entering our side
                # Only the border column of our side is possible
                distribution[self.all_columns_except_border_of_our_team, :] = 0
            elif not self.was_pacman and not is_pacman:
                # Enemy was a ghost and is still a ghost, should be on their side
                # Only columns of their side are possible
                distribution[self.our_side_columns, :] = 0
            elif self.was_pacman and is_pacman:
                # Enemy was Pacman and is still Pacman, should be on our side
                # Only columns of our side are possible
                distribution[self.enemy_side_columns, :] = 0
            
            # normalize 
            if distribution.sum() > 0:
                distribution /= distribution.sum()
            else:
                # TODO: if np.sum is 0 for all distr in history just return the initial distr
                print(f'oh no, distribution for enemy {self.tracked_enemy_index} is completely zero')# TODO remove
                print(f'{self.was_pacman=} {is_pacman=}')
                print(f'{old_distribution=}')
                raise "WTF"

        # Update the was_pacman state for the next iteration
        self.was_pacman = is_pacman


    def __get_condensed_position_distribution(self):
        """
        Returns a condensed version of all distributions in self.noisy_position_distributions.
        """
        # These way the noise hopefully cancels out
        condensed_distribution = np.sum(self.noisy_position_distributions, axis=0)
        # normalize
        condensed_distribution /= np.sum(self.noisy_position_distributions)
        return condensed_distribution
        
    def __weigh_particles(self, condensed_position_distribution):
        """
        Weigh particles based on the condensed position distribution.
        """
        int_particles = np.rint(self.particles).astype(int)
        self.weights[:] *= condensed_position_distribution[int_particles[:,0], int_particles[:,1]]
        self.weights /= self.weights.sum()

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
        self.velocities[:num_resampled_particles] = self.velocities[indices]
        self.weights[:num_resampled_particles] = self.weights[indices]
 
        # Add random particles
        # TODO try constraining random particles to be near to 
        # the current estimated position? -> set prob to 0 elsewhere in condensed_position_distribution before drawing?
        flat_distribution = condensed_position_distribution.flatten()
        random_indices = np.random.choice(np.arange(len(flat_distribution)), size=num_random_particles, p=flat_distribution)
        self.particles[num_resampled_particles:] = np.array(np.unravel_index(random_indices, condensed_position_distribution.shape)).T
        self.velocities[num_resampled_particles:] = self.__generate_random_velocities(self.particles[num_resampled_particles:])
        self.weights[num_resampled_particles:] = flat_distribution[random_indices]

    def __generate_random_velocities(self, current_positions, current_velocities=None):
        SPEED = 1.0 # TODO change if enemy is scared -> just one bool per enemy should be enough. needs to be reset if scared ghost is eaten. also, use normal speed when scared but on the pacman side for each particle.
        TURN_OR_MOVEMENT_START_PROBABILITY = 0.3 # TODO tune these values (or maybe even learn them). or analyze from historical game recordings.
        REVERSE_PROBABILITY = 0.1
        STOP_PROBABILITY = 0.05

        velocities = np.zeros((len(current_positions), 2))

        for i, current_position in enumerate(current_positions):
            current_direction = Actions.vector_to_direction(current_velocities[i]) if current_velocities is not None else Directions.STOP 
            config = Configuration(current_position, current_direction)
            possible_actions = Actions.get_possible_actions(config, self.walls)
            # Action is chosen randomly, but not uniformly:
            # - stop action is least probable
            # - reversing direction is more probable
            # - turning or starting moving is even more probable
            # - continuing moving in the same direction is most probable

            # Calculate probabilities for each action
            action_probabilities = np.zeros(len(possible_actions))
            for action_index, action in enumerate(possible_actions):
                prev_velocity = self.velocities[action_index]
                new_velocity  = np.array(Actions.direction_to_vector(action, SPEED)) # TODO: use scared ghost speed when applicable (particle in enemy half & enemy has scared timer)

                if np.array_equal(new_velocity, (0, 0)):
                    action_probabilities[action_index] = STOP_PROBABILITY
                elif _is_reverse(new_velocity, prev_velocity):
                    action_probabilities[action_index] = REVERSE_PROBABILITY
                elif _is_turn_or_movement_start(new_velocity, prev_velocity):
                    action_probabilities[action_index] = TURN_OR_MOVEMENT_START_PROBABILITY
                else:
                    action_probabilities[action_index] = 1 - TURN_OR_MOVEMENT_START_PROBABILITY - REVERSE_PROBABILITY - STOP_PROBABILITY

            # Normalize probabilities
            action_probabilities /= action_probabilities.sum()
            
            # Choose action based on probabilities
            action = np.random.choice(possible_actions, p=action_probabilities)
            dx, dy = Actions.direction_to_vector(action, SPEED) # TODO: use scared ghost speed when applicable (particle in enemy half & enemy has scared timer)
            velocities[i] = [dx, dy]

        return velocities


    def __is_valid(self, position):
        """
        Checks if position is valid for an agent (i.e. it is not inside a wall).
        """
        x, y = np.rint(position).astype(int)
        return not self.walls[x][y]
    
    def __get_adjacent_cells(self, x, y):
        """
        Get adjacent cells to the given cell (x, y), considering the walls.
        """
        adjacent_cells = []
        for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.walls.width and 0 <= new_y < self.walls.height and not self.walls[new_x][new_y]:
                adjacent_cells.append((new_x, new_y))
        return adjacent_cells



def _is_turn_or_movement_start(new_velocity, prev_velocity):
    """
    Check if the change from prev_velocity to new_velocity represents a 90-degree turn,
    accounting for possible changes in speed due to the scared flag of an agent changing.

    Also returns True when prev_velocity was not moving and new_velocity is moving.
    """
    # Normalize the velocities to unit vectors for direction comparison
    prev_direction = prev_velocity / np.linalg.norm(prev_velocity) if np.linalg.norm(prev_velocity) != 0 else prev_velocity
    new_direction = new_velocity / np.linalg.norm(new_velocity) if np.linalg.norm(new_velocity) != 0 else new_velocity

    # A 90-degree turn occurs if the direction changes in one axis but remains constant in the other
    return (prev_direction[0] == new_direction[0] and prev_direction[1] != new_direction[1]) or \
           (prev_direction[1] == new_direction[1] and prev_direction[0] != new_direction[0])

def _is_reverse(new_velocity, prev_velocity):
    """
    Check if the change from prev_velocity to new_velocity represents a reversal of direction,
    accounting for possible changes in speed due to the scared flag of an agent changing.
    """
    # Normalize the velocities to unit vectors for direction comparison
    prev_direction = prev_velocity / np.linalg.norm(prev_velocity) if np.linalg.norm(prev_velocity) != 0 else prev_velocity
    new_direction = new_velocity / np.linalg.norm(new_velocity) if np.linalg.norm(new_velocity) != 0 else new_velocity

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
                raise 'WTF?!' # pups TODO don't throw exception and just let it go on
                j = N - 1

    if n < N:
        return np.random.choice(indices, size=n, replace=False)
    
    return indices