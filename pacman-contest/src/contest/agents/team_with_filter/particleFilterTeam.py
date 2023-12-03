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
from contest.game import Directions
from contest.util import nearestPoint, manhattanDistance, Counter
from contest.agents.team_with_filter.particleFilter import EnemyPositionParticleFilter
from contest.agents.team_with_filter.ownFoodSupervisor import OwnFoodSupervisor
import numpy as np
import logging
from contest.agents.team_with_filter.customLogging import *
from contest.agents.team_with_filter.enemySuicideDetector import EnemySuicideDetector

# Needs to be initialized in CaptureAgent.register_initial_state
# pups TODO: could put this in some instance of class CommunicationModule, that one instance can be shared by both our agents
enemyPositionParticleFilters = dict()


np.seterr(all='raise')


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

# TODO create the objects in create_team and 
# pass in the enemies to the constructor of the agents instead of using global vars
ownFoodSupervisor = OwnFoodSupervisor()
enemySuicideDetector = EnemySuicideDetector()

# {
#   <enemy_index>: [<friendly agent indices who saw the enemy>]
# }
lastSeenEnemies = {}

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
        self.estimated_distances_logger.addHandler(DeferredFileHandler(f'estimated_distances_agent_{self.index}'))
        self.estimated_distances_logger.setLevel(logging.DEBUG)

        self.true_distances_logger = logging.getLogger(f'true distances {self.index}')
        self.true_distances_logger.addHandler(DeferredFileHandler(f'true_distances_{self.index}'))
        self.true_distances_logger.setLevel(logging.DEBUG)

        self.noisy_distances_logger = logging.getLogger(f'noisy distances {self.index}')
        self.noisy_distances_logger.addHandler(DeferredFileHandler(f'noisy_distances_{self.index}'))
        self.noisy_distances_logger.setLevel(logging.DEBUG)


    def writeLogFiles(self):
        for handler in [*self.estimated_distances_logger.handlers, *self.true_distances_logger.handlers, *self.noisy_distances_logger.handlers]:
            if type(handler) is DeferredFileHandler:
                handler.flush()


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        # remember who the real enemy is
        self.enemies = self.get_opponents(game_state)
        self.totalAgents = len(game_state.teams)



        # initialize enemy position particle filters
        global enemyPositionParticleFilters
        if not enemyPositionParticleFilters:
            for enemy in self.enemies:
                enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(num_particles=500, 
                                               walls=game_state.get_walls(), 
                                               initial_position=game_state.get_agent_position(enemy),
                                               tracked_enemy_index=enemy)
        global ownFoodSupervisor
        if not ownFoodSupervisor.initialized:
            own_food = self.get_food_you_are_defending(game_state)
            own_capsules = self.get_capsules_you_are_defending(game_state)
            ownFoodSupervisor.initialize(own_food, own_capsules, self.totalAgents)

        global enemySuicideDetector
        if not enemySuicideDetector.initialized:
            team_spawn_positions = {agentOnTeam: game_state.get_initial_agent_position(agentOnTeam) for agentOnTeam in self.agentsOnTeam}
            enemySuicideDetector.initialize(self.enemies, self.agentsOnTeam, team_spawn_positions)


    def get_exact_opponent_distances(self, game_state): 
        agent_pos = game_state.get_agent_position(self.index)
        enemy_positions = [game_state.get_agent_position(i) for i in self.enemies]
        return [manhattanDistance(agent_pos, enemy_pos) if enemy_pos is not None else None for enemy_pos in enemy_positions]


    def get_noisy_opponent_distances(self, game_state):
        distances = game_state.get_agent_distances()
        return [distances[i] for i in self.enemies]


    def update_particle_filter(self, game_state):
        """
        - Moves the particles of the preceding enemy's filter
        - Updates particle weights and resamples particles of every enemy's filter with exact position or noisy distance estimate
        """ 
        global ownFoodSupervisor
        ownFoodSupervisor.update(self.index, self.get_food_you_are_defending(game_state), self.get_capsules_you_are_defending(game_state))

        # IMPORTANT! Suicide Detector must be updated before particle filter!
        global enemySuicideDetector
        enemySuicideDetector.update(game_state)

        global enemyPositionParticleFilters

        # Update the particle filter of the preceding enemy
        # only if it's not the very first move of the game (i.e. 1200 steps are left)
        if game_state.data.timeleft != 1200:
            # Determine the index of the enemy who just moved
            enemy_who_just_moved = (self.index - 1) % self.totalAgents
            # move particles of filter one time step into the future
            enemyPositionParticleFilters[enemy_who_just_moved].move_particles()

        # TODO update isPacman state of particle filter
        
        # for all enemies, update particle filter with exact position or noisy distance
        agent_position = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        for enemy_index in self.enemies:
            pf = enemyPositionParticleFilters[enemy_index]
            enemy_is_pacman = game_state.get_agent_state(index=enemy_index).is_pacman
            # try getting an exact position and update with the exact position
            exact_pos = game_state.get_agent_position(enemy_index)
            if exact_pos is not None:
                self.logger.info(f'Got exact position of enemy {enemy_index} at {exact_pos}!')
                pf.update_with_exact_position(exact_pos, enemy_is_pacman)
            # if enemy agent committed suicide, reset him to his spawn
            elif enemySuicideDetector.hasCommittedSuicide(enemy_index):
                if game_state.data._eaten[enemy_index]:
                    raise 'wtf is this' # TODO remove, it works now
                pf.reset_to_spawn()
            elif ownFoodSupervisor.canLocalizeEnemy(enemy_index):
                assert enemy_is_pacman, "He should be if he just ate ..." # TODO remove
                pf.update_with_exact_position(ownFoodSupervisor.localizeEnemy(), is_pacman=True)
            # if enemy agent is not seen, update particle filter with noisy distance
            else:
                pf.update_with_noisy_distance(agent_position, noisy_distances[enemy_index], enemy_is_pacman)


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

        self.update_particle_filter(game_state)
        
        global enemyPositionParticleFilters
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
        for enemy in self.enemies:
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
        
        #return Directions.STOP
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
