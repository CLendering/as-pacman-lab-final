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

from contest.captureAgents import CaptureAgent
from enemy_localization.particleFilter import EnemyPositionParticleFilter
from contest.util import manhattanDistance
from collections import defaultdict
from enemy_localization.customLogging import *
import numpy as np


class ParticleFilterAgent(CaptureAgent):
    """Base class for agents using a particle filter and other detectors to localize enemies"""
    def __init__(self, index, enemyPositionParticleFilters, ownFoodSupervisor, enemySuicideDetector, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.enemyPositionParticleFilters = enemyPositionParticleFilters
        self.ownFoodSupervisor = ownFoodSupervisor
        self.enemySuicideDetector = enemySuicideDetector


        if EnemyPositionParticleFilter._LOGGING:        
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
        if EnemyPositionParticleFilter._LOGGING:
            for handler in [*self.estimated_distances_logger.handlers, *self.true_distances_logger.handlers, *self.noisy_distances_logger.handlers]:
                if type(handler) is DeferredFileHandler:
                    handler.flush()

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)

        # remember who the real enemy is
        self.enemies = self.get_opponents(game_state)
        self.totalAgents = len(game_state.teams)

        # initialize enemy position particle filters
        if not self.enemyPositionParticleFilters:
            for enemy in self.enemies:
                self.enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(
                                               num_particles=500, 
                                               noisy_position_distribution_buffer_length=10,
                                               walls=game_state.get_walls(), 
                                               initial_position=game_state.get_agent_position(enemy),
                                               tracked_enemy_index=enemy)
        if not self.ownFoodSupervisor.initialized:
            own_food = self.get_food_you_are_defending(game_state)
            own_capsules = self.get_capsules_you_are_defending(game_state)
            self.ownFoodSupervisor.initialize(own_food, own_capsules, self.totalAgents)

        if not self.enemySuicideDetector.initialized:
            team_spawn_positions = {agentOnTeam: game_state.get_initial_agent_position(agentOnTeam) for agentOnTeam in self.agentsOnTeam}
            self.enemySuicideDetector.initialize(self.enemies, self.agentsOnTeam, team_spawn_positions)

    def update_particle_filter(self, game_state):
        """
        Updates particle filter incorporating the information from the EnemySuicideDetector and the OwnFoodSupervisor
        """ 
        self.ownFoodSupervisor.update(self.index, self.get_food_you_are_defending(game_state), self.get_capsules_you_are_defending(game_state))

        # suicide Detector must be updated before particle filter!
        self.enemySuicideDetector.update(game_state)

        # Update the particle filter of the preceding enemy (unless we got the very first move of the game)
        enemy_who_just_moved = game_state.data._agent_moved
        if enemy_who_just_moved is not None:
            # move particles of filter one time step into the future
            self.enemyPositionParticleFilters[enemy_who_just_moved].move_particles()

        
        # for all enemies, update particle filter with exact position or noisy distance
        self.own_position = game_state.get_agent_position(self.index)
        noisy_distances = game_state.get_agent_distances()
        for enemy_index in self.enemies:
            pf = self.enemyPositionParticleFilters[enemy_index]
            enemy_is_pacman = game_state.get_agent_state(index=enemy_index).is_pacman
            # try getting an exact position and update with the exact position
            exact_pos = game_state.get_agent_position(enemy_index)
            if exact_pos is not None:
                pf.update_with_exact_position(exact_pos, enemy_is_pacman)
            # if enemy agent committed suicide, reset him to his spawn
            elif self.enemySuicideDetector.hasCommittedSuicide(enemy_index):
                pf.reset_to_spawn()
            elif self.ownFoodSupervisor.canLocalizeEnemy(enemy_index):
                pf.update_with_exact_position(self.ownFoodSupervisor.localizeEnemy(), is_pacman=True)
            # if enemy agent is not seen, update particle filter with noisy distance
            else:
                pf.update_with_noisy_distance(self.own_position, noisy_distances[enemy_index], enemy_who_just_moved, enemy_is_pacman)



        # LOGGING
        if EnemyPositionParticleFilter._LOGGING:
            estimated_pos = self.get_distinct_enemy_position_estimates()
            DEBUG_actual_enemy_positions = game_state.DEBUG_actual_enemy_positions
            self.estimated_distances_logger.info(self.get_distinct_enemy_distance_estimates())
            self.true_distances_logger.info(game_state.DEBUG_actual_enemy_distances)
            self.noisy_distances_logger.info([noisy_distances[enemy] for enemy in self.enemies])
            for i, enemy in enumerate(self.enemies):
                pf = self.enemyPositionParticleFilters[enemy]
                pf.estimated_positions_logger.info(estimated_pos[i])
                pf.true_positions_logger.info(DEBUG_actual_enemy_positions[i])
            if game_state.data.timeleft < 4:
                # last turn of agent: write own log files
                self.writeLogFiles()
                if game_state.data.timeleft < 2:
                # last agent of team: flush particle filter logs
                    for pf in self.enemyPositionParticleFilters.values():
                        pf.writeLogFiles()

    def get_distinct_enemy_position_estimates(self):
        """
        Get a list of estimated enemy positions.
        Estimates the position for each enemy as the mean of the particle distribution.
        Returns a single position for each enemy.        
        """
        return [self.enemyPositionParticleFilters[enemy].estimate_distinct_position() for enemy in sorted(self.enemyPositionParticleFilters.keys())]
    
    def get_distinct_enemy_distance_estimates(self):
        """
        Get a list of estimated enemy distances.
        Estimates the position for each enemy as the mean of the particle distribution
        and calculates the distance from that.
        Returns a single distance for each enemy.        
        """
        distinct_enemy_position_estimates = self.get_distinct_enemy_position_estimates()
        return [manhattanDistance(self.own_position, enemy_position) for enemy_position in distinct_enemy_position_estimates]

    def get_probabilistic_enemy_position_estimates(self):
        """
        Get probabilistic position estimates.
        Returns a array of positions where the value at each position is the probability of that position.
        """
        return [self.enemyPositionParticleFilters[enemy].estimate_probabilistic_position() for enemy in sorted(self.enemyPositionParticleFilters.keys())]
    
    def get_probabilistic_enemy_distance_estimates(self):
        """
        Generate probabilistic distance estimates.
        Returns a list of dicts (1 dict for each enemy).
        Each dict maps a distance to a probability.
        """
        # List of dicts for each enemy
        # Each dict maps a distance to a probability
        probabilistic_distance_estimates = [defaultdict(int) for _ in range(len(self.enemies))]
        
        for i_enemy, probabilistic_enemy_position_estimate in enumerate(self.get_probabilistic_enemy_position_estimates()):
            x, y = probabilistic_enemy_position_estimate.nonzero()

            # Distance x Probability matrix, each entry is a tuple (distance, probability)
            distance_probability_distribution = np.empty((len(x), len(x)), dtype="i,f") # distance is int. probability is float
            # Fill matrix with manhattan distances and probabilities
            for i, enemy_position in enumerate(zip(x, y)):
                probability = probabilistic_enemy_position_estimate[enemy_position]
                distance = manhattanDistance(self.own_position, enemy_position)
                probabilistic_distance_estimates[i_enemy][distance] += probability
            
        return probabilistic_distance_estimates