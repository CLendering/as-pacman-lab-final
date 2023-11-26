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
from contest.capture import SONAR_NOISE_RANGE, SONAR_NOISE_VALUES
import numpy as np
import filterpy.kalman as kf

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

class EnemyPositionParticleFilter:
    # TODO maybe I need to add some noise to some other part here?
    def __init__(self, num_particles, walls, initial_position):
        # dtype needs to be float because scared ghosts move slower (0.5), normal speed is 1.0 
        self.particles = np.full((num_particles, 2), initial_position, dtype=float)
        self.weights = np.full(num_particles, 1/num_particles)
        self.num_particles = num_particles

        self.walls = walls
        self.spawn_position = initial_position
        self.previous_velocities = np.zeros((num_particles, 2), dtype=float)  # Initialize with zero velocity

    def move_particles(self):
        """
        Move particles within the allowed range and considering the map's walls.
        Should be called exactly once after the enemy has actually moved. (e.g. agent 2 calls this in his turn to update enemy 1's position)
        """
        SPEED = 1.0 # TODO change if enemy is scared. needs to be reset if scared ghost is eaten. also, use normal speed when scared but on the pacman side.
        TURN_OR_MOVEMENT_START_PROBABILITY = 0.3 # TODO tune these values (or maybe even learn them)
        REVERSE_PROBABILITY = 0.1
        STOP_PROBABILITY = 0.05

        #action_counter = Counter()

        for particle_index, p in enumerate(self.particles):
            x, y = p
            dummy_config = Configuration((x, y), 'North')
            possible_actions = Actions.get_possible_actions(dummy_config, self.walls)
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
                prev_velocity = self.previous_velocities[action_index]
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
            dx, dy = Actions.direction_to_vector(action, SPEED)

            #action_counter[action] +=1
            
            # Move particles
            self.particles[particle_index] = x + dx, y + dy
            self.previous_velocities[particle_index] = [dx, dy]
        #print(action_counter)
    
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

    def update_with_noisy_distance(self, measured_distance, agent_position):
        """
        Recalculates weights and resamples particles with the given noisy distance.
        Every agent should call this in every turn with all noisy estimates they get.

        If an agent gets an exact position estimate, 
        they should not call this method but update_with_exact_position instead.
        """
        self.__weigh_particles(measured_distance, agent_position)
        self.__resample_particles()

    def estimate_position(self):
        """
        Estimate the position as the mean of the particle distribution.
        """
        mean_position = np.mean(self.particles, axis=0)
        # Check if mean position is valid
        if self.__is_valid(mean_position):
            return mean_position
        
        # Find the average of the nearest valid particles to the mean position
        distances = np.linalg.norm(self.particles - mean_position, ord=1, axis=1)
        min_distance = np.min(distances)
        nearest_particles = self.particles[distances == min_distance]
        mean_position = np.mean(nearest_particles, axis=0)

        # Check if mean position is valid now
        if self.__is_valid(mean_position):
            return mean_position
        
        # mean position is still not valid - just chose a random particle for now.
        # TODO save last estimates and in this case pick the particle closest to last estimates
        mean_position = np.random.choice(nearest_particles)
        return mean_position
    
    def __is_valid(self, position):
        """
        Checks if position is valid for an agent (i.e. it is not inside a wall).
        """
        x, y = position
        x_int, y_int = int(x + 0.5), int(y + 0.5)
        return not self.walls[x_int][y_int]

    def __weigh_particles(self, measured_distance, agent_position):
        """
        Weigh particles based on how closely they match the measured distance.
        """
        self.latest_noisy_distance = measured_distance
        self.latest_agent_position = agent_position
        for i in range(self.num_particles):
            # Calculate the predicted Manhattan distance from the particle to the agent
            predicted_distance = np.sum(np.abs(self.particles[i] - agent_position))

            # Compare with the measured distance and check its likelihood 
            # given that we know the noise is a discrete uniform distribution 
            # of the values in `SONAR_NOISE_VALUES`
            observation_diff = predicted_distance - measured_distance

            # Check if the difference is within the possible range of noise values
            if observation_diff <= max(SONAR_NOISE_VALUES):
                # Count how many noise values can align the prediction with the actual observation
                # this can be one of:
                # - 0 (difference is not within plausible range of noise values)
                # - 1 (noise is 0)
                # - 2 (noise is +x or -x, x > 0)
                valid_noise_values = np.sum(np.abs(SONAR_NOISE_VALUES) >= observation_diff)
                # Assign weight proportional to the number of valid noise values
                self.weights[i] = valid_noise_values / SONAR_NOISE_RANGE
            else:
                # If the difference is too large (i.e. can't be explained by the sonar noise),
                #  the weight for this particle is zero
                self.weights[i] = 0
        
        # Normalize the weights so it's a probability distribution (will be used to resample particles)
        if sum(self.weights) == 0:
            self.weights += 1.e-300  # avoid divide by zero
        self.weights /= sum(self.weights)
        
    def __resample_particles(self):
       """
       Resample particles with replacement based on their weights.
       """
       indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
       self.particles = self.particles[indices]

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


### Kalman Filter
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, measurement_noise):
        """
        Initialize the Kalman filter.

        :param initial_state: Initial estimate of the state.
        :param initial_covariance: Initial estimate of the error covariance.
        :param measurement_noise: Variance of the measurement noise.
        """
        self.state_estimate = initial_state
        self.error_covariance = initial_covariance
        self.measurement_noise = measurement_noise
        self.measurement_matrix = np.eye(len(initial_state))  # Assuming the same dimensions for state and measurements

    def predict(self, process_noise):
        """
        Predict step of the Kalman filter.

        :param process_noise: The noise covariance to add for prediction uncertainty.
        """
        # Predict the state estimate (assuming it remains constant in this case)
        # state_estimate = state_estimate (no change)

        # Predict the error covariance
        self.error_covariance += process_noise

    def update(self, measurement):
        """
        Update step of the Kalman filter.

        :param measurement: The new measurement to update the state.
        """
        # Compute Kalman Gain
        kalman_gain = self.error_covariance.dot(self.measurement_matrix.T).dot(
            np.linalg.inv(self.measurement_matrix.dot(self.error_covariance).dot(self.measurement_matrix.T) + self.measurement_noise))

        # Update the state estimate
        self.state_estimate += kalman_gain.dot(measurement - self.measurement_matrix.dot(self.state_estimate))

        # Update the error covariance
        identity_matrix = np.eye(len(self.state_estimate))
        self.error_covariance = (identity_matrix - kalman_gain.dot(self.measurement_matrix)).dot(self.error_covariance)

    def get_estimated_state(self):
        """
        Get the current estimated state.
        """
        return self.state_estimate



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


history_enemy_position_estimates = []
history_DEBUG_actual_enemy_positions = []
history_enemy_distance_estimates = []
history_DEBUG_actual_enemy_distances = []
history_noisy_distances = []

pf_estimated_positions_f = open('particle_filter/estimated_positions.log', 'w')
pf_estimated_distances_f = open('particle_filter/estimated_distances.log', 'w')
true_positions_f = open('particle_filter/true_positions.log', 'w')
true_distances_f = open('particle_filter/true_distances.log', 'w')
noisy_distances_f = open('particle_filter/noisy_distances.log', 'w')


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
    

        #self.mses_noisy_f = open('kalman_filter/mse_noisy.log', 'w')
        #self.mses_kf_f = open('kalman_filter/mse_kf.log', 'w')
        self.kf_estimated_distances_f = open(f'kalman_filter/estimated_distances_agent_{index}.log', 'w')
        self.kf_true_distances_f = open(f'kalman_filter/true_distances_agent_{index}.log', 'w')
        self.kf_noisy_distances_f = open(f'kalman_filter/noisy_distances_agent_{index}.log', 'w')

        self.move_number = 0

        self.history_kf_estimated_distances = []
        self.history_kf_true_distances = []     
        self.history_kf_noisy_distances = [] 

    def __del__(self):
        if self.index == max(self.agentsOnTeam):
            s_pf_estimated_positions_f = ''
            s_true_positions_f = ''
            s_pf_estimated_distances_f = ''
            s_true_distances_f = ''
            s_noisy_distances_f = ''
            for i in range(len(history_enemy_position_estimates)):
                s_pf_estimated_positions_f += f'{tuple(history_enemy_position_estimates[i][0])} {tuple(history_enemy_position_estimates[i][1])}\n'
                s_true_positions_f += f'{history_DEBUG_actual_enemy_positions[i][0]} {history_DEBUG_actual_enemy_positions[i][1]}\n'
                s_pf_estimated_distances_f += f'{history_enemy_distance_estimates[i][0]} {history_enemy_distance_estimates[i][1]}\n'
                s_true_distances_f += f'{history_DEBUG_actual_enemy_distances[i][0]} {history_DEBUG_actual_enemy_distances[i][0]}\n'
                s_noisy_distances_f += f'{history_noisy_distances[i][0]} {history_noisy_distances[i][1]}\n'

            pf_estimated_positions_f.write(s_pf_estimated_positions_f)
            true_positions_f.write(s_true_positions_f)
            pf_estimated_distances_f.write(s_pf_estimated_distances_f)
            true_distances_f.write(s_true_distances_f)
            noisy_distances_f.write(s_noisy_distances_f)

        s_kf_estimated_distances = ''
        s_kf_noisy_distances = ''
        s_kf_true_distances = ''
        for i in range(self.move_number):
            e = self.history_kf_estimated_distances[i]
            n = self.history_kf_noisy_distances[i]
            t = self.history_kf_true_distances[i]
            s_kf_estimated_distances += f'{e[0]} {e[1]}\n'
            s_kf_noisy_distances += f'{n[0]} {n[1]}\n'
            s_kf_true_distances += f'{t[0]} {t[1]}\n'
        self.kf_estimated_distances_f.write(s_kf_estimated_distances)
        self.kf_noisy_distances_f.write(s_kf_noisy_distances)
        self.kf_true_distances_f.write(s_kf_true_distances)



        pf_estimated_positions_f.close()
        pf_estimated_distances_f.close()
        true_positions_f.close()
        true_distances_f.close()
        noisy_distances_f.close()

        #self.mses_noisy_f.close()
        #self.mses_kf_f.close()
        self.kf_estimated_distances_f.close()
        self.kf_noisy_distances_f.close()
        self.kf_true_distances_f.close()


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # initialize enemy position particle filters
        global enemyPositionParticleFilters
        if not enemyPositionParticleFilters:
            for enemy in self.get_opponents(game_state):
                enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(num_particles=1000, 
                                               walls=game_state.get_walls(), 
                                               initial_position=game_state.get_agent_position(enemy)) 
        self.finishedFirstMove = False
                                        
        # register_initial_state is called once at the very beginning of a game
        # here we can get the enemies exact initial starting position
        # and use this information to initialize the kalman filter for estimating the enemy distances
        enemy_positions = [game_state.get_agent_position(enemy) for enemy in self.get_opponents(game_state)]
        agent_position = game_state.get_agent_position(self.index)
        enemy_distances = [manhattanDistance(agent_position, enemy_position) for enemy_position in enemy_positions]
        initial_covariance = np.eye(len(enemy_distances)) * 0.1 # smaller value due to high certainty of initial enemy positions
        SONAR_NOISE_RANGE = 13 # from capture.py: noise is from [-6, 6]
        # https://proofwiki.org/wiki/Variance_of_Discrete_Uniform_Distribution
        # can use formula thanks to shift-invariance https://math.stackexchange.com/a/456866
        enemy_distances_variance = (SONAR_NOISE_RANGE**2 - 1) / 12
        
        kalman_filter =  kf.KalmanFilter (dim_x=2, dim_z=2)
        # assign initial state of enemy distances
        kalman_filter.x = enemy_distances
        # initial state is known exactly -> covariance matrix of that state is 0
        kalman_filter.P *= 0
        # define the state transition matrix (for now, assume that distance in the next step is the same as in the previous step)
        kalman_filter.F = np.eye(len(enemy_distances))
        # measurement function
        kalman_filter.H = np.eye(len(enemy_distances))
        # measurement noise
        self.noisy_measurement_noise = np.eye(len(enemy_distances)) * enemy_distances_variance
        kalman_filter.R = self.noisy_measurement_noise
        # process noise
        var = (5**2 - 1) / 12 # distance can change at most +-2 -> variance of discrete uniform distribution [-2, 2] == var of [1, 5] (shift invariance)
        kalman_filter.Q = np.eye(len(enemy_distances)) * var


        self.enemy_distances_kalman_filter = kalman_filter

        #self.enemy_distances_kalman_filter = KalmanFilter(enemy_distances, initial_covariance, enemy_distances_variance)


    def get_exact_opponent_distances(self, game_state): 
        agent_pos = game_state.get_agent_position(self.index)
        enemy_positions = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
        return [manhattanDistance(agent_pos, enemy_pos) if enemy_pos is not None else None for enemy_pos in enemy_positions]


    def get_noisy_opponent_distances(self, game_state):
        distances = game_state.get_agent_distances()
        return [distances[i] for i in self.get_opponents(game_state)]

    def update_enemy_distances_kalman_filter(self, game_state):
        # Get noisy measurements as a fallback
        new_measurements_noisy = np.array(self.get_noisy_opponent_distances(game_state))
        # Attempt to get exact measurements
        new_measurements_exact = self.get_exact_opponent_distances(game_state)

        # Prepare an array to store the final measurements for the update
        final_measurements = new_measurements_noisy.copy()

        # Iterate over the measurements to check for exact values
        for i, exact_measurement in enumerate(new_measurements_exact):
            if exact_measurement is not None:
                # If an exact measurement is available, use it and adjust R for that measurement
                final_measurements[i] = exact_measurement
                # Set a lower R value (indicating higher confidence) for this measurement
                self.enemy_distances_kalman_filter.R[i, i] = 0 # Set an appropriate lower value
            else:
                # If no exact measurement, use the noisy measurement and the default R value
                self.enemy_distances_kalman_filter.R[i, i] = self.noisy_measurement_noise[i, i]

        # Perform the predict and update steps
        self.enemy_distances_kalman_filter.predict()
        self.enemy_distances_kalman_filter.update(final_measurements)

        return self.enemy_distances_kalman_filter.x


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.move_number += 1

        PRINT = False
        actions = game_state.get_legal_actions(self.index)
        if PRINT: print(f"I'm agent {self.index}")
        noisy_distances = self.get_noisy_opponent_distances(game_state)
        # make the comparison fair - kalman filter also gets exact distances when possible
        exact_distances = self.get_exact_opponent_distances(game_state)
        for i, d in enumerate(exact_distances):
            if d is not None:
                noisy_distances[i] = d


        agent_position = game_state.get_agent_position(self.index)

        global enemyPositionParticleFilters

        enemies = self.get_opponents(game_state)
        # move particle filter one time step into the future
        # for the enemy who played just before our agent
        enemy_who_just_moved = (self.index - 1) % len(game_state.teams)
        # but only move particles if the enemy player actually moved (i.e. if I have the first move in the game, don't update enemy's particle filter)
        if not self.finishedFirstMove or self.index > enemy_who_just_moved:
            enemyPositionParticleFilters[enemy_who_just_moved].move_particles()
        
        # update particle filter with exact position or noisy distance
        for enemy_index, noisy_distance in zip(enemies, noisy_distances):
            pf = enemyPositionParticleFilters[enemy_index]
            # try getting an exact position
            exact_pos = game_state.get_agent_position(enemy_index)
            if exact_pos is not None:
                pf.update_with_exact_position(exact_pos)
            else:
                pf.update_with_noisy_distance(noisy_distance, agent_position)

        # get new estimates of enemy positions
        enemy_position_estimates = [enemyPositionParticleFilters[enemy].estimate_position() for enemy in sorted(enemyPositionParticleFilters.keys())]
        enemy_distance_estimates = [manhattanDistance(agent_position, enemy_pos) for enemy_pos in enemy_position_estimates]

        self.finishedFirstMove = True
        DEBUG_actual_enemy_positions = game_state.DEBUG_actual_enemy_positions

        # TODO hier sind paar sachen ganz komisch.
        # ich glaub beim setzen von DEBUG_actual_enemy_distances geht was schief
        # z.b. in der aktuellen datei bei row 100 schwingen die werte zu sehr rum das geht gar nich
        # die estimated und noisy schwingen auch extrem ka was da los is

        # just for evaluating the kalman filter: get the actual positions of the enemies
        DEBUG_actual_enemy_distances = game_state.DEBUG_actual_enemy_distances




        history_enemy_position_estimates.append(enemy_position_estimates)
        history_DEBUG_actual_enemy_positions.append(DEBUG_actual_enemy_positions)
        history_enemy_distance_estimates.append(enemy_distance_estimates)
        history_DEBUG_actual_enemy_distances.append(DEBUG_actual_enemy_distances)
        history_noisy_distances.append(noisy_distances)

        kf_estimated_distances = self.update_enemy_distances_kalman_filter(game_state)
        rounded_estimated_distances = [round(d) for d in kf_estimated_distances]
        self.history_kf_noisy_distances.append(noisy_distances)
        self.history_kf_true_distances.append(DEBUG_actual_enemy_distances)
        self.history_kf_estimated_distances.append(kf_estimated_distances)

        if PRINT: print(f'{noisy_distances=}')
        if PRINT: print(f'kalman_filter_distances={rounded_estimated_distances} ({kf_estimated_distances})')
        if PRINT: print(f'true_distances={DEBUG_actual_enemy_distances}')

        error_noisy_distances = np.array(DEBUG_actual_enemy_distances) - np.array(noisy_distances)
        MSE_noisy_distances = (error_noisy_distances**2).mean()
        if PRINT: print(f'{MSE_noisy_distances=}')
        error_kalman_filter_distances = np.array(DEBUG_actual_enemy_distances) - np.array(rounded_estimated_distances)
        MSE_kalman_filter_distances = (error_kalman_filter_distances**2).mean()
        if PRINT: print(f'{MSE_kalman_filter_distances=}')

        #if self.move_number == 300:
        #    self.mses_noisy_f.write(f'{MSE_noisy_distances}\n')
        #    self.mses_kf_f.write(f'{MSE_kalman_filter_distances}\n')



        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

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
            return best_action

        return random.choice(best_actions)

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
