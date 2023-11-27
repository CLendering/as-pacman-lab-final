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
from contest.pacman import GhostRules
import numpy as np
from collections import deque
import logging

# ---- Logging ----
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

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indices = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices

class EnemyPositionParticleFilter:
    # TODO maybe I need to add some noise to some other part here?
    def __init__(self, num_particles, walls, initial_position, tracked_enemy_index, max_noisy_estimates=10):
        self.logger = logging.getLogger(f'EnemyPositionParticleFilter (enemy {tracked_enemy_index})')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_log_handler)

        self.recent_noisy_estimates = deque([] * max_noisy_estimates ,maxlen=max_noisy_estimates)

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


    def move_particles(self):
        """
        Move particles within the allowed range and considering the map's walls.
        Should be called exactly once after the enemy has actually moved. (e.g. agent 2 calls this in his turn to update enemy 1's position)
        """
        self.logger.info(f'Moving particles')
        current_particles_counter = Counter()
        for pos in self.particles:
            current_particles_counter[tuple(pos)] += 1
        self.logger.info(f'Current particle positions: {current_particles_counter}')


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
        self.logger.info(f'New particle positions: {new_particles_counter}')
    
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
        self.recent_noisy_estimates.append(measured_distance)
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
       Resample particles based on their weights.
       """
       # multinomial resample (with replacement) based on weights
       #indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
       indices = systematic_resample(self.weights)
       # TODO use systematic resample instead of multinomial resample
       self.particles = self.particles[indices]
       self.velocities = self.velocities[indices]

    # TODO continue here
    def __generate_recent_random_particles(self, num_particles):
        """
        Generate random particles with a preference for positions closer to recent noisy estimates.
        """
        recent_random_particles = []
        num_estimates = len(self.recent_noisy_estimates)

        # Create a probability distribution based on proximity to noisy estimates
        probabilities = np.zeros(len(self.legal_positions))
        for index, noisy_estimate in enumerate(self.recent_noisy_estimates):
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

        self.logger = logging.getLogger(f'Agent {self.index})')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_log_handler)
    

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


        pf_estimated_positions_f.close()
        pf_estimated_distances_f.close()
        true_positions_f.close()
        true_distances_f.close()
        noisy_distances_f.close()



    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # initialize enemy position particle filters
        global enemyPositionParticleFilters
        if not enemyPositionParticleFilters:
            for enemy in self.get_opponents(game_state):
                enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(num_particles=1000, 
                                               walls=game_state.get_walls(), 
                                               initial_position=game_state.get_agent_position(enemy),
                                               tracked_enemy_index=enemy) 
        self.finishedFirstMove = False
        
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
            # if enemy agent is not seen directly, update particle filter with noisy distance
            else:
                pf.update_with_noisy_distance(noisy_distances[enemy_index], agent_position)


    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        self.logger.info(f"Turn starts")

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

        self.finishedFirstMove = True
        DEBUG_actual_enemy_positions = game_state.DEBUG_actual_enemy_positions

        # TODO hier sind paar sachen ganz komisch.
        # ich glaub beim setzen von DEBUG_actual_enemy_distances geht was schief
        # z.b. in der aktuellen datei bei row 100 schwingen die werte zu sehr rum das geht gar nich
        # die estimated und noisy schwingen auch extrem ka was da los is

        # just for evaluating the filter: get the actual positions of the enemies
        DEBUG_actual_enemy_distances = game_state.DEBUG_actual_enemy_distances




        history_enemy_position_estimates.append(enemy_position_estimates)
        history_DEBUG_actual_enemy_positions.append(DEBUG_actual_enemy_positions)
        history_enemy_distance_estimates.append(enemy_distance_estimates)
        history_DEBUG_actual_enemy_distances.append(DEBUG_actual_enemy_distances)
        history_noisy_distances.append(noisy_distances)


        error_noisy_distances = np.array(DEBUG_actual_enemy_distances) - np.array(noisy_distances)
        MSE_noisy_distances = (error_noisy_distances**2).mean()


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
