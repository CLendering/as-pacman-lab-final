from enemy_localization.customLogging import logging, console_log_handler 
import numpy as np


class OwnFoodSupervisor():
    """
    Keeps track of the food and capsules that are eaten by our enemies to localize them.
    """
    def __init__(self):
        self.initialized = False
        self.logger = logging.getLogger('OFS)')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(console_log_handler)
    
    def initialize(self, initial_own_food, initial_own_capsules, total_agents):
        if self.initialized:
            self.logger.critical('OFS.initialize was called although it was already initialized before')
        self.initialized = True
        self.lastOwnFood = np.array(initial_own_food.data) # initial_own_food is of type contest.Game.grid -> data property has list data
        self.lastOwnCapsules = set(initial_own_capsules) # initial_own_capsules is a list of tuples
        self.totalAgents = total_agents

        self._localized_enemy_index = None
        self._localized_enemy_position = None

    def update(self, agent_index, own_food, own_capsules):
        # Reset localized enemy information
        self._localized_enemy_index = None
        self._localized_enemy_position = None
    
        own_food = np.array(own_food.data)
        own_capsules = set(own_capsules)

        # Some of our food was eaten! (But do nothing if enemy was killed and dropped food on our side)
        if np.any(self.lastOwnFood.sum() > own_food.sum()):
            # x and y indices of food that is gone now 
            # (both np.arrays of length 1 because only one of our own pellets can go missing each time it's our turn) 
            x, y = (self.lastOwnFood != own_food).nonzero()
            self._localized_enemy_position = (x[0], y[0])
            self._localized_enemy_index = (agent_index - 1) % self.totalAgents
        elif self.lastOwnCapsules != own_capsules:
            differenceCapsules = list(self.lastOwnCapsules.difference(own_capsules))
            if len(differenceCapsules) == 1:
                self._localized_enemy_position = differenceCapsules[0]
                self._localized_enemy_index = (agent_index - 1) % self.totalAgents
            else:
                self.logger.critical('Wtf? last={self.lastOwnCapsules}, now={own_capsules}, d={differenceCapsules}')
    
        # Update own food and capsules
        np.copyto(self.lastOwnFood, own_food)
        self.lastOwnCapsules = own_capsules
    
    
    def canLocalizeEnemy(self, index):
        return self._localized_enemy_index is index

    def localizeEnemy(self):
        return self._localized_enemy_position
