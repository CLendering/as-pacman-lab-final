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
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
from queue import PriorityQueue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='BetterReflexAgent', second='DefensiveReflexAgent', num_training=0):
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

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        print(game_state)
        actions = game_state.get_legal_actions(self.index)

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
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}
    
    
class BetterReflexAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
    
    def choose_action(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)

        # Use A* algorithm to find the best action
        best_action = self.astar_search(game_state)

        return best_action
    

    def get_actions(goal_state, parents):
        """
        Retrieves the sequence of actions taken to reach the given goal state.
        
        Parameters:
        - goal_state: The goal state.
        - parents: A dictionary mapping a state to its parent state and the action taken to reach it.

        Returns:
        - A list of actions leading to the goal state.
        """

        # Initialize the list of actions
        actions_list = []
        
        # Start with the goal state
        cur_state = goal_state

        # Trace back the actions taken to reach the goal state
        while True:
            # Break the loop if the current state is not in the parents dictionary
            if cur_state not in parents:
                break
            # Otherwise, update the current state and append the action taken to reach it
            cur_state, act = parents[cur_state]
            actions_list.append(act)

        # Reverse the action list to get the correct order from start to goal
        actions_list.reverse()
        return actions_list

    def evaluate(self, game_state, action):
    
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = game_state.generate_successor(agent_index=self.index, action=action)

        if self.red:
            newFood = successorGameState.get_red_food()
            team_indexes = successorGameState.red_team
            enemy_indexes = successorGameState.blue_team
        else:
            newFood = successorGameState.get_blue_food()
            team_indexes = successorGameState.blue_team
            enemy_indexes = successorGameState.red_team

        newPos = successorGameState.get_agent_position(index=self.index)
        
        newGhostStates = []
        for enemy_index in enemy_indexes:
            enemy_state = successorGameState.get_agent_state(index=enemy_index)
            if enemy_state.configuration is not None:
                if not enemy_state.is_pacman:
                    newGhostStates.append(enemy_state)

        newScaredTimes = [ghostState.scared_timer for ghostState in newGhostStates]

        new_agent_state = successorGameState.get_agent_state(index=self.index)
        previous_agent_state = game_state.get_agent_state(index=self.index)



        
        # Our approach was to compute a score based on the distance to the closest food, the number of remaining foods, 
        # the distance to ghosts, and the remaining scared timer for the ghosts, using weights that were determined by trial and error.

        # We found that the weights that worked best were:
        # -5 for the number of remaining foods
        # -8 for the distance to the closest food
        # 7 for the distance to ghosts
        # 1 for the remaining scared timer for the ghosts

        # And the final score was computed as:
        # closest_food_dist_weight * closest_food_dist + \
        # food_count_weight * food_count + \
        # ghost_manhattan_distance_weight * ghost_manhattan_distance + \
        # ghost_scared_timer_weight * ghost_scared_timer

        # Calculate the previous food count before the action
        # Calculate the previous food count before the action
        new_food_count = new_agent_state.num_carrying
        prev_food_count = previous_agent_state.num_carrying

      
        # Calculate the Manhattan distance from Pacman to each remaining food pellet
        food_manhattan_distances = []
        for food_pos in newFood.as_list():
            food_manhattan_distances.append(abs(food_pos[0] - newPos[0]) + abs(food_pos[1] - newPos[1]))

        
        # Define weights for the distance to the closest food and calculate the distance to the closest food, by
        # taking the minimum of the Manhattan distances to each remaining food pellet
        closest_food_dist_weight = -8
        closest_food_dist = min(food_manhattan_distances)
        
        # If Pacman has eaten a food pellet in the proposed action, set the closest food distance weight to 0 so it doesn't affect the score
        if prev_food_count != new_food_count:
            closest_food_dist_weight = 0
        
        # Set the ghost manhattan distance weight
        ghost_manhattan_distance_weight = 7

        # Calculate the Manhattan distance from Pacman to each ghost
        ghost_manhattan_distance = 0
        for ghost_state in newGhostStates:
            ghost_pos = ghost_state.get_position()
            dist = abs(ghost_pos[0] - newPos[0]) + abs(ghost_pos[1] - newPos[1])
            # If a ghost is on Pacman's position on the proposed action, set a very negative value (-infinity) so Pacman avoids it
            if dist == 0:
                ghost_manhattan_distance = -float('inf')
                break
            ghost_manhattan_distance += dist
        
        # If a ghost is on Pacman's position, return the -infinity score immediately - Pacman should avoid this action
        if ghost_manhattan_distance == -float('inf'):
            return ghost_manhattan_distance
        
        # If Pacman has finished eating all food pellets in the proposed action, set an infinite reward so Pacman will take this
        # action and win the game
        if closest_food_dist == float('inf'):
            return float('inf')
        
        # Define weights for the remaining scared timer for the ghosts and calculate the remaining scared timer for the ghosts
        # as the sum of the scared timers for each ghost
        ghost_scared_timer_weight = 1
        ghost_scared_timer = sum(newScaredTimes)
        
        # Return the final score as the weighted sum of the closest food distance, the number of remaining foods, the ghost Manhattan distance,
        # and the remaining scared timer for the ghosts
        return closest_food_dist_weight * closest_food_dist + \
               ghost_manhattan_distance_weight * ghost_manhattan_distance + \
               ghost_scared_timer_weight * ghost_scared_timer

    def choose_action(self, game_state):
        """Search the node that has the lowest combined cost and heuristic first."""

        actions = game_state.get_legal_actions(self.index)
        
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = min(values)
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

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
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
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

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
        features = util.Counter()
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
