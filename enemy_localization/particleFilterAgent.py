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


        if LOGGING_ENABLED:        
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
        if LOGGING_ENABLED:
            for handler in [*self.estimated_distances_logger.handlers, *self.true_distances_logger.handlers, *self.noisy_distances_logger.handlers]:
                if type(handler) is DeferredFileHandler:
                    handler.flush()

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)

        self.own_position = game_state.get_agent_position(self.index)
        # remember who the real enemy is
        self.enemies = self.get_opponents(game_state)
        self.totalAgents = len(game_state.teams)

        # initialize enemy position particle filters if not initialized yet or if we are in a new game
        if not self.enemyPositionParticleFilters or any(not pf.initializedFor(game_state) for pf in self.enemyPositionParticleFilters.values()):
            for enemy in self.enemies:
                self.enemyPositionParticleFilters[enemy] = EnemyPositionParticleFilter(
                                               num_particles=500, 
                                               noisy_position_distribution_buffer_length=10,
                                               initial_game_state=game_state,
                                               tracked_enemy_index=enemy)
        self.manhattan_distance_grid = self.enemyPositionParticleFilters[self.enemies[0]].manhattan_distance_grid
                
        if not self.ownFoodSupervisor.initializedFor(game_state):
            own_food = self.get_food_you_are_defending(game_state)
            own_capsules = self.get_capsules_you_are_defending(game_state)
            self.ownFoodSupervisor.initialize(own_food, own_capsules, self.totalAgents, game_state)

        if not self.enemySuicideDetector.initializedFor(game_state):
            team_spawn_positions = {agentOnTeam: game_state.get_initial_agent_position(agentOnTeam) for agentOnTeam in self.agentsOnTeam}
            self.enemySuicideDetector.initialize(self.enemies, self.agentsOnTeam, team_spawn_positions, game_state)

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
        if LOGGING_ENABLED:
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
        Get a dict of estimated enemy positions.
        Estimates the position for each enemy as the mean of the particle distribution.
        For each enemy, contains a single position for each enemy.        
        """
        return {enemy: self.enemyPositionParticleFilters[enemy].estimate_distinct_position() for enemy in self.enemyPositionParticleFilters}
    
    def get_distinct_enemy_distance_estimates(self):
        """
        Get a dict of estimated enemy distances.
        Estimates the position for each enemy as the mean of the particle distribution
        and calculates the Manhattan distance from that.
        For each enemy, contains a single distance for each enemy.        
        """
        distinct_enemy_position_estimates = self.get_distinct_enemy_position_estimates()
        return {enemy: self.manhattan_distance_grid[self.own_position[0], self.own_position[1], enemy_position[0], enemy_position[1]] for enemy, enemy_position in distinct_enemy_position_estimates.items()}

    def get_probabilistic_enemy_position_estimates(self):
        """
        Get a dict of probabilistic position estimates.
        For each enemy, contains a array of positions where the value at each position is the probability of that position.
        """
        return {enemy: self.enemyPositionParticleFilters[enemy].estimate_probabilistic_position() for enemy in self.enemyPositionParticleFilters}
    
    def get_probabilistic_enemy_distance_estimates(self):
        """
        Generate probabilistic distance estimates.
        Returns a dict of dicts (1 dict for each enemy).
        For each enemy, contains a dict which maps a Manhattan distance to a probability.
        """
        # List of dicts for each enemy
        # Each dict maps a distance to a probability
        probabilistic_distance_estimates = {enemy: defaultdict(float) for enemy in self.enemies}
        
        for enemy, probabilistic_enemy_position_estimate in self.get_probabilistic_enemy_position_estimates().items():
            x, y = probabilistic_enemy_position_estimate.nonzero()

            # Fill matrix with manhattan distances and probabilities
            for enemy_position in zip(x, y):
                probability = probabilistic_enemy_position_estimate[enemy_position]
                distance = self.manhattan_distance_grid[self.own_position[0], self.own_position[1], enemy_position[0], enemy_position[1]]
                probabilistic_distance_estimates[enemy][distance] += probability
            
        return probabilistic_distance_estimates