# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
from contest.captureAgents import CaptureAgent
from contest.game import Directions, Configuration, Actions
from contest.util import nearestPoint, manhattanDistance, Counter
from contest.capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES, SIGHT_RANGE
from contest.pacman import GhostRules
import numpy as np
from collections import deque
import logging
import os

# ---- Logging ----
def createEmptyLogDir(baseDir):
    if not os.path.exists(baseDir):
        os.makedirs(baseDir)

    if not os.listdir(baseDir):  # Check if directory is empty
        return baseDir

    # Find the next available directory name
    counter = 1
    while True:
        new_dir = os.path.join(baseDir, f'{counter:02d}')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            break
        counter += 1

    return new_dir

logDir = createEmptyLogDir('particle_filter')

def getLogFilePath(filename):
    return os.path.join(logDir, filename)





class DeferredFileHandler(logging.Handler):
    def __init__(self, name, formatter=logging.Formatter('%(message)s')):
        super().__init__()
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)
        self.name = name
        self.buffer = []

    def emit(self, record):
        self.buffer.append(self.format(record))

    def flush(self):
        with open(f'{self.name}.log', 'a') as f:
            f.write('\n'.join(self.buffer) + '\n')
        self.buffer = []



console_log_handler = logging.StreamHandler()  # Console handler
file_log_handler = logging.FileHandler('particleFilterTeam.log')  # File handler
console_log_handler.setLevel(logging.INFO)
file_log_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_log_handler.setFormatter(c_format)
file_log_handler.setFormatter(f_format)



# Needs to be initialized in CaptureAgent.register_initial_state
# TODO: could put this in some instance of class CommunicationModule, that one instance can be shared by both our agents
enemyPositionParticleFilters = dict()


def is_turn_or_movement_start(new_velocity, prev_velocity):
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

def is_reverse(new_velocity, prev_velocity):
    """
    Check if the change from prev_velocity to new_velocity represents a reversal of direction,
    accounting for possible changes in speed due to the scared flag of an agent changing.
    """
    # Normalize the velocities to unit vectors for direction comparison
    prev_direction = prev_velocity / np.linalg.norm(prev_velocity) if np.linalg.norm(prev_velocity) != 0 else prev_velocity
    new_direction = new_velocity / np.linalg.norm(new_velocity) if np.linalg.norm(new_velocity) != 0 else new_velocity

    # A reversal occurs if the new direction is the opposite of the previous one
    return np.array_equal(new_direction, -prev_direction)

def systematic_resample(weights, n=None):
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
                raise 'WTF?!' # TODO don't throw exception and just let it go on
                j = N - 1

    if n < N:
        return np.random.choice(indices, size=n, replace=False)
    
    return indices

# TODO check runtime of particle filter, add way to scale down or completely 
# shut down if it takes too much time and we risk defaulting game bc of that (1 sec max per turn I think)
class EnemyPositionParticleFilter:
    # TODO maybe I need to add some noise to some other part here?
    def __init__(self, num_particles, walls, initial_position, tracked_enemy_index, max_noisy_estimates=10):
        self.logger = logging.getLogger(f'EnemyPositionParticleFilter (enemy {tracked_enemy_index})')
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(console_log_handler)

        # Distributions based on noisy measurements, not actual particle positions
        # probability distributions of possible positions calculated from noisy distance estimates and agent position
        # this is used as further input to update the belief (particles) of the particle filter
        initial_position_distribution = np.zeros((walls.width, walls.height))
        initial_position_distribution[initial_position] = 1
        self.noisy_position_distributions = deque([np.copy(initial_position_distribution) for _ in range(max_noisy_estimates)], maxlen=max_noisy_estimates)

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


        self.estimated_positions_logger = logging.getLogger(f'estimated positions enemy {tracked_enemy_index})')
        self.estimated_positions_logger.addHandler(DeferredFileHandler(getLogFilePath(f'estimated_positions_enemy_{tracked_enemy_index}')))
        self.estimated_positions_logger.setLevel(logging.DEBUG)

        self.true_positions_logger = logging.getLogger(f'true positions enemy {tracked_enemy_index})')
        self.true_positions_logger.addHandler(DeferredFileHandler(getLogFilePath(f'true_positions_enemy_{tracked_enemy_index}')))
        self.true_positions_logger.setLevel(logging.DEBUG)

    def writeLogFiles(self):
        for handler in [*self.estimated_positions_logger.handlers, *self.true_positions_logger.handlers]:
            if type(handler) is DeferredFileHandler:
                handler.flush()


    def move_particles(self):
        """
        Move particles within the allowed range and considering the map's walls.
        Should be called exactly once after the enemy has actually moved. (e.g. agent 2 calls this in his turn to update enemy 1's position)
        """
        self.logger.info(f'Moving particles')
        current_particles_counter = Counter()
        for pos in self.particles:
            current_particles_counter[tuple(pos)] += 1
        
        sorted_particle_positions = sorted(current_particles_counter.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f'Current top 10 particle positions: {dict(sorted_particle_positions[:10])}')


        SPEED = 1.0 # TODO change if enemy is scared. needs to be reset if scared ghost is eaten. also, use normal speed when scared but on the pacman side.
        TURN_OR_MOVEMENT_START_PROBABILITY = 0.3 # TODO tune these values (or maybe even learn them)
        REVERSE_PROBABILITY = 0.1
        STOP_PROBABILITY = 0.05

        action_counter = Counter()

        for particle_index, ((x, y), v) in enumerate(zip(self.particles, self.velocities)):
            config = Configuration((x, y), Actions.vector_to_direction(v)) # or use 'STOP'?
            possible_actions = Actions.get_possible_actions(config, self.walls)
            # TODO: make action selection better by not drawing uniformly but
            # - making stop action way less probable
            # - increasing probability of actions which have the same direction as the previous action (-> so save velocity vector in another np.array)
            # - making actions which are opposite direction of previous action a little bit less probable
            #    -> save previous particles, calculate movement vectors (p_t - p_{t-1}) 
            #    -> compare last movement vector to new movement vector (dx, dy)
            #    -> use dot product to rate probability of the movement
            #  
            # - make stopping way less probable 

            # Calculate probabilities for each action
            action_probabilities = np.zeros(len(possible_actions))
            for action_index, action in enumerate(possible_actions):
                prev_velocity = self.velocities[action_index] # TODO hier ist ein bug
                new_velocity  = np.array(Actions.direction_to_vector(action, SPEED)) # TODO: use scared ghost speed when applicable (particle in enemy half & enemy has scared timer)

                # TODO check comparison actually works or do I need to convert to np array & use np comparison methods
                if np.array_equal(new_velocity, (0, 0)):
                    action_probabilities[action_index] = STOP_PROBABILITY
                elif is_reverse(new_velocity, prev_velocity):
                    action_probabilities[action_index] = REVERSE_PROBABILITY
                elif is_turn_or_movement_start(new_velocity, prev_velocity):
                    action_probabilities[action_index] = TURN_OR_MOVEMENT_START_PROBABILITY
                else:
                    action_probabilities[action_index] = 1 - TURN_OR_MOVEMENT_START_PROBABILITY - REVERSE_PROBABILITY - STOP_PROBABILITY

            # Normalize probabilities
            action_probabilities /= action_probabilities.sum()
            
            # Choose action based on probabilities
            action = np.random.choice(possible_actions, p=action_probabilities)
            dx, dy = Actions.direction_to_vector(action, SPEED) # TODO: use scared ghost speed when applicable (particle in enemy half & enemy has scared timer)

            action_counter[action] +=1
            
            # Update particle position and velocity
            self.particles[particle_index] = x + dx, y + dy
            self.velocities[particle_index] = [dx, dy]
        
        self.logger.info(f'Selected actions: {action_counter}')
        new_particles_counter = Counter()
        for pos in self.particles:
            new_particles_counter[tuple(pos)] += 1

        new_sorted_particle_positions = sorted(new_particles_counter.items(), key=lambda x: x[1], reverse=True)
        self.logger.info(f'New particle positions: {dict(new_sorted_particle_positions[:10])}')
    
    # TODO call update_with_exact_position with enemy spawn when killing an enemy
    def update_with_exact_position(self, position):
        """
        Sets all particles to an exactly known position.
        Call this when:
        - An enemy agent is directly seen and we get their exact position.
        - An enemy agent has been killed. They'll respawn at their spawn point,
          so we can set all particles to that position (saved in EnemyPositionParticleFilter.spawn_position)
        """
        self.particles[:] = position
        self.weights[:] = 1/self.num_particles

        position_distribution = np.zeros((self.walls.width, self.walls.height))
        position_distribution[position] = 1
        for i in range(len(self.noisy_position_distributions)):
            self.noisy_position_distributions[i] = np.copy(position_distribution)

    def reset_to_spawn(self):
        """
        Call this when an enemy has been killed to reset 
        the particles and position distributions to their initial position.
        """
        self.update_with_exact_position(self.spawn_position)

    def update_with_noisy_distance(self, agent_position, noisy_distance):
        """
        Recalculates weights and resamples particles with the given noisy distance.
        Every agent should call this in every turn with all noisy estimates they get.

        If an agent gets an exact position estimate, 
        they should not call this method but update_with_exact_position instead.
        """
        self.__update_noisy_position_distributions(agent_position, noisy_distance)
        condensed_position_distribution = self.__get_condensed_position_distribution()
        self.__weigh_particles(condensed_position_distribution)
        self.__resample_particles(condensed_position_distribution)

    def estimate_position(self):
        """
        Estimate the position as the mean of the particle distribution.
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


    def __update_noisy_position_distributions(self, agent_position, noisy_distance):
        """
        Update the noisy position distributions based on a new noisy distance measurement.
        """
        # TODO add method which sets position probability of cells within sight range of agent to zero


        # make old position distributions fade into adjacent cells
        self.__flatten_noisy_position_distributions()

        # add new position distribution
        self.__add_new_noisy_position_distribution(agent_position, noisy_distance)

        # reset probabilities of positions in SIGHT_RANGE to zero
        self.__clear_probabilities_in_sight_range(agent_position)
 
    def __flatten_noisy_position_distributions(self):
        """
        Flatten out the noisy position distributions to account for potential movement as time progresses.
        """
        # Each agent (enemy or friendly) moves once every four turns. They can move at most 1 grid cell per turn. The particle filters for every enemy is updated every two turns. So every time a particle filter is updated, the enemy moves on average by +-0.5  (because it can move by 1 every four turns and the particle filters are updated every two turns). The same goes for the measuring agent. So the true distance of a measuring agent to every enemy can change by |+-0.5|+|+-0.5|=1 every two turns. The position of an enemy agent can change by one of the vectors in [(-0.5, 0), (0, -0.5), (0, 0), (0.5, 0), (0,0.5)] every two turns. 
        # Remember that The position of an enemy agent can change by one of the vectors in [(-0.5, 0), (0, -0.5), (0, 0), (0.5, 0), (0,0.5)] every two turns when the particle filters are being updated.

        # TODO do this next
        # TODO I'm not sure if this is the best idea, probabilities might oscillate back and forth
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
                print('oo')

    def __within_sight_range(self, agent_pos, pos):
        """
        Check if pos is within SIGHT_RANGE of agent_pos.
        """
        return np.linalg.norm(np.array(agent_pos) - np.array(pos), ord=1) <= SIGHT_RANGE


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
        indices = systematic_resample(self.weights, num_resampled_particles)
        self.particles[:num_resampled_particles] = self.particles[indices]
        self.velocities[:num_resampled_particles] = self.velocities[indices]
        self.weights[:num_resampled_particles] = self.weights[indices]
 
        # Add random particles
        # TODO try constraining random particles to be near to 
        # the current estimated position? -> set prob to 0 elsewhere in condensed_position_distribution before drawing?
        flat_distribution = condensed_position_distribution.flatten()
        random_indices = np.random.choice(np.arange(len(flat_distribution)), size=num_random_particles, p=flat_distribution)
        self.particles[num_resampled_particles:] = np.array(np.unravel_index(random_indices, condensed_position_distribution.shape)).T
        self.velocities[num_resampled_particles:] = self.__generate_random_velocities(num_random_particles)
        self.weights[num_resampled_particles:] = flat_distribution[random_indices]

    def __generate_random_velocities(self, num_random_particles):
        # TODO implement
        # For the generation of the velocities of random particles, the velocity should be calculated such that:
        # - the velocity is legal (use Actions.get_possible_actions(Configuration(position, Directions.STOP), walls) to check this)
        # - most velocities should point towards the current most probable position in condensed_position_distribution
        return np.full((num_random_particles, 2), (0,0), dtype=float)


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


    # TODO continue here (but need to rewrite this bc noisy position estimates is now prob distr)
    def __generate_recent_random_particles(self, num_particles):
        """
        Generate random particles with a preference for positions closer to recent noisy estimates.
        """
        recent_random_particles = []
        num_estimates = len(self.noisy_position_distributions)

        # Create a probability distribution based on proximity to noisy estimates
        probabilities = np.zeros(len(self.legal_positions))
        for index, noisy_estimate in enumerate(self.noisy_position_distributions):
            for i, pos in enumerate(self.legal_positions):
                distance = np.sum(np.abs(pos - self.agent_position))
                if distance == noisy_estimate:
                    probabilities[i] += (num_estimates - index)  # Higher weight for newer estimates

        # Normalize probabilities
        probabilities /= probabilities.sum()

        # Choose random positions based on the probability distribution
        chosen_indices = np.random.choice(len(self.legal_positions), size=num_particles, p=probabilities)
        recent_random_particles = self.legal_positions[chosen_indices]

        return recent_random_particles



    # TODO: use some random particles to fight particle deprivation
    # def __resample_particles(self):
    #     """
    #     Resample particles with replacement based on their weights, 
    #     while also introducing some random but probable particles based on the current noisy distance.
    #     """
    #     RANDOM_PARTICLES_PERCENTAGE = 0.001  # e.g., 0.1% random particles
    #     num_random_particles = int(self.num_particles * RANDOM_PARTICLES_PERCENTAGE)

    #     # Resample based on weights for the majority of particles
    #     indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles - num_random_particles, p=self.weights)
    #     resampled_particles = self.particles[indices]

    #     # Add random but probable particles
    #     probable_random_particles = self.__generate_probable_random_particles(num_random_particles)

    #     # Combine resampled and probable random particles
    #     self.particles = np.vstack((resampled_particles, probable_random_particles))

    # def __generate_probable_random_particles(self, num_particles):
    #     """
    #     Generate random particles that are probable given the current noisy distance estimate.
    #     """
    #     probable_random_particles = []
    #     while len(probable_random_particles) < num_particles:
    #         # Generate a random position within a plausible range
    #         x = np.random.randint(0, self.walls.width)   # Adjust range according to your game area
    #         y = np.random.randint(0, self.walls.height)  # Adjust range according to your game area
    #         if not self.walls[x][y]:  # Check if the position is not a wall
    #             distance = np.sum(np.abs(np.array([x, y]) - self.latest_agent_position))  # Manhattan distance
    #             if self.latest_noisy_distance - 6 <= distance <= self.latest_noisy_distance + 6:  # Check if distance is within the range of noisy estimate
    #                 probable_random_particles.append([x, y])
    #     return np.array(probable_random_particles)


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

        self.logger = logging.getLogger(f'Agent {self.index})')
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(console_log_handler)
    

        self.estimated_distances_logger = logging.getLogger(f'estimated distances {self.index}')
        self.estimated_distances_logger.addHandler(DeferredFileHandler(getLogFilePath(f'estimated_distances_agent_{self.index}')))
        self.estimated_distances_logger.setLevel(logging.DEBUG)

        self.true_distances_logger = logging.getLogger(f'true distances {self.index}')
        self.true_distances_logger.addHandler(DeferredFileHandler(getLogFilePath(f'true_distances_{self.index}')))
        self.true_distances_logger.setLevel(logging.DEBUG)

        self.noisy_distances_logger = logging.getLogger(f'noisy distances {self.index}')
        self.noisy_distances_logger.addHandler(DeferredFileHandler(getLogFilePath(f'noisy_distances_{self.index}')))
        self.noisy_distances_logger.setLevel(logging.DEBUG)


        self.last_food_you_are_defending = None


    def writeLogFiles(self):
        for handler in [*self.estimated_distances_logger.handlers, *self.true_distances_logger.handlers, *self.noisy_distances_logger.handlers]:
            if type(handler) is DeferredFileHandler:
                handler.flush()


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # initialize enemy position particle filters
        global enemyPositionParticleFilters
        if not enemyPositionParticleFilters:
            for enemy in self.get_opponents(game_state):
                enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(num_particles=500, 
                                               walls=game_state.get_walls(), 
                                               initial_position=game_state.get_agent_position(enemy),
                                               tracked_enemy_index=enemy) 
        
    def get_exact_opponent_distances(self, game_state): 
        agent_pos = game_state.get_agent_position(self.index)
        enemy_positions = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
        return [manhattanDistance(agent_pos, enemy_pos) if enemy_pos is not None else None for enemy_pos in enemy_positions]


    def get_noisy_opponent_distances(self, game_state):
        distances = game_state.get_agent_distances()
        return [distances[i] for i in self.get_opponents(game_state)]


    def update_particle_filter(self, game_state):
        """
        - Moves the particles of the preceding enemy's filter
        - Updates particle weights and resamples particles of every enemy's filter with exact position or noisy distance estimate
        """
        global enemyPositionParticleFilters

        # Update the particle filter of the preceding enemy
        # only if it's not the very first move of the game (i.e. 1200 steps are left)
        if game_state.data.timeleft != 1200:
            # Determine the index of the enemy who just moved
            enemy_who_just_moved = (self.index - 1) % len(game_state.teams)
            # move particles of filter one time step into the future
            enemyPositionParticleFilters[enemy_who_just_moved].move_particles()
        
        # for all enemies, update particle filter with exact position or noisy distance
        agent_position = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        for enemy_index in self.get_opponents(game_state):
            pf = enemyPositionParticleFilters[enemy_index]
            # try getting an exact position and update with the exact position
            exact_pos = game_state.get_agent_position(enemy_index)
            if exact_pos is not None:
                self.logger.info(f'Got exact position of enemy {enemy_index} at {exact_pos}!')
                pf.update_with_exact_position(exact_pos)
                
                # TODO update with exact position spawn when killing enemy
                #if GhostRules.canKill()
            # if enemy agent is not seen directly, update particle filter with noisy distance
            else:
                pf.update_with_noisy_distance(agent_position, noisy_distances[enemy_index])
            
            # Logging
            # TODO


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.logger.info(f"Turn starts")



        # bf=game_state.get_blue_food()
        # rf=game_state.get_red_food()
        # bc=game_state.get_blue_capsules()
        # rc=game_state.get_red_capsules()
        # c = game_state.get_capsules()
        # hasfood= game_state.has_food(0,0)

        # food= self.get_food(game_state)
        # own_food = self.get_food_you_are_defending(game_state)
        # capsules = self.get_capsules(game_state)
        # own_capsules = self.get_capsules_you_are_defending(game_state)

        # TODO: do same with capsules 

        # TODO do this check in a class that is shared by both agents (so it can update the particle filters more frequently)
        # and then update pf of enemy with closest position to missing food 
        food_you_are_defending = self.get_food_you_are_defending(game_state)
        if self.last_food_you_are_defending is not None:
            if self.last_food_you_are_defending != food_you_are_defending:
                enemy_positions = (np.array(self.last_food_you_are_defending.data) != np.array(food_you_are_defending.data)).nonzero()
                # tuple with 2 elements
                #   -> 1st element: np.array of x indices of food that is gone now
                #   -> 2nd element: np.array of y indices of food that is gone now
                # TODO LET'S GOOO HERE
                for enemy in self.get_opponents(game_state):
                    # TODO use information of how num_carrying for each enemy changes
                    # to find out which enemy is at the position of the missing food :)
                    # TODO: figure out direction of enemy as well from this lol
                    # probably easiest by getting the closest direction from this vector:
                    # (missing food position - last enemy position estimate)
                    
                    game_state.get_agent_state(index=enemy).num_carrying


        self.last_food_you_are_defending = food_you_are_defending
            

        actions = game_state.get_legal_actions(self.index)
        noisy_distances = self.get_noisy_opponent_distances(game_state)
        # make the comparison fair - filter also gets exact distances when possible
        exact_distances = self.get_exact_opponent_distances(game_state)
        for i, d in enumerate(exact_distances):
            if d is not None:
                noisy_distances[i] = d


        agent_position = game_state.get_agent_position(self.index)

        global enemyPositionParticleFilters

        self.update_particle_filter(game_state)

        enemies = self.get_opponents(game_state)
        
        # get new estimates of enemy positions
        enemy_position_estimates = [enemyPositionParticleFilters[enemy].estimate_position() for enemy in sorted(enemyPositionParticleFilters.keys())]
        enemy_distance_estimates = [manhattanDistance(agent_position, enemy_pos) for enemy_pos in enemy_position_estimates]



        # TODO delete this game_state.DEBUG_... stuff (just for evaluating the filter)
        # just for evaluating the filter: get the actual positions of the enemies
        # LOGGING
        DEBUG_actual_enemy_positions = game_state.DEBUG_actual_enemy_positions
        self.estimated_distances_logger.info(enemy_distance_estimates)
        self.true_distances_logger.info(game_state.DEBUG_actual_enemy_distances)
        self.noisy_distances_logger.info(noisy_distances)
        for i, enemy in enumerate(sorted(enemyPositionParticleFilters.keys())):
            pf = enemyPositionParticleFilters[enemy]
            pf.estimated_positions_logger.info(enemy_position_estimates[i])
            pf.true_positions_logger.info(DEBUG_actual_enemy_positions[i])


        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        chosen_action = None
        
        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            chosen_action = best_action
        else:
            chosen_action = random.choice(best_actions)
        
        # Check if the chosen action results in eating an enemy
        # in that case reset the enemy's position to their spawn
        successor = self.get_successor(game_state, chosen_action)
        for enemy in enemies:
            if successor.data._eaten[enemy]:
                enemyPositionParticleFilters[enemy].reset_to_spawn()


        # LOGGING
        if game_state.data.timeleft < 4:
            # last turn of agent: write own log files
            self.writeLogFiles()
            if game_state.data.timeleft < 2:
            # last agent of team: flush particle filter logs
                for pf in enemyPositionParticleFilters.values():
                    pf.writeLogFiles()
        
        return Directions.STOP
        return chosen_action
        

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        
        # agent_distances = game_state.get_agent_distances()
        # print(f'{game_state}')
        # print(f'{agent_distances=}')

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
