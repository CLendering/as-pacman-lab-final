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



class ParticleFilterAgent(CaptureAgent):
    """Base class for agents using a particle filter and other detectors to localize enemies"""
    def __init__(self, index, enemyPositionParticleFilters, ownFoodSupervisor, enemySuicideDetector, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.enemyPositionParticleFilters = enemyPositionParticleFilters
        self.ownFoodSupervisor = ownFoodSupervisor
        self.enemySuicideDetector = enemySuicideDetector

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
                                               noisy_distances_buffer_length=10,
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

        # Update the particle filter of the preceding enemy
        # unless it's the very first move of the game (i.e. 1200 steps are left)
        if game_state.data.timeleft != 1200:
            # Determine the index of the enemy who just moved
            enemy_who_just_moved = (self.index - 1) % self.totalAgents
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
                pf.update_with_noisy_distance(self.own_position, noisy_distances[enemy_index], enemy_is_pacman)

    def get_enemy_position_estimates(self):
        return [self.enemyPositionParticleFilters[enemy].estimate_position() for enemy in sorted(self.enemyPositionParticleFilters.keys())]
    
    def get_enemy_distance_estimates(self):
        enemy_position_estimates = self.get_enemy_position_estimates()
        return [manhattanDistance(self.own_position, enemy_position) for enemy_position in enemy_position_estimates]
