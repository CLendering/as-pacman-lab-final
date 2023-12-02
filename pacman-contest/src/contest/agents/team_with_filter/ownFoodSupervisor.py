# TODO implement
from contest.agents.team_with_filter.customLogging import *
import numpy as np


class OwnFoodSupervisor():
    def __init__(self):
        self.initialized = False
        self.logger = logging.getLogger('OwnFoodSupervisor)')
        self.logger.setLevel(logging.WARNING)
        self.logger.addHandler(console_log_handler)
    
    def initialize(self, initial_own_food, initial_own_capsules, enemy_indices):
        if self.initialized:
            self.logger.critical('OwnFoodSupervisor.initialize was called although it was already initialized before')
        self.initialized = True
        self.last_own_food = np.array(initial_own_food.data) # initial_own_food is of type contest.Game.grid -> data property has list data
        self.last_own_capsules = set(initial_own_capsules) # initial_own_capsules is a list of tuples
        self.enemy_indices = enemy_indices
        self.enemies_carried_food = {enemy: 0 for enemy in enemy_indices}

    def update(self, own_food, own_capsules):
        # TODO do this check in a class that is shared by both agents (so it can update the particle filters more frequently)
        # and then update pf of enemy with closest position to missing food 
        own_food = np.array(own_food.data)
        if np.any(self.last_own_food != own_food):
            enemy_positions = (self.last_own_food != own_food).nonzero()
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
                    # TODO also use is_pacman to find out which side they're on:
                    # TODO do the same with how num_carrying/num_returned is changing
                    # TODO actually no - that is redundant. looking at how is_pacman changes is enough
                    # is_pacman at T | is_pacman at T + 1 | info
                    # False          | False              | on their side
                    # False          | True               | just entered our side (exactly at 1 column)
                    # True           | False              | just entered their side (exactly at 1 column)
                    # True           | True               | on our side
                    # and when num_returned increases, we know they just arrived on their side (so we're sure they're in 1 column)

        # Update food array
        np.copyto(self.last_own_food, own_food)
