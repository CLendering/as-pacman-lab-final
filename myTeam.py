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

from abc import abstractmethod
import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
from queue import PriorityQueue
from contest.game import Directions, Configuration, Actions
from contest.capture import AgentRules

#################
# Utility #
#################
def get_theoretical_legal_successors(position, game_state):
        """
        Returns the legal actions for an agent
        """
        dummy_config = Configuration(position, 'North')
        possible_actions = Actions.get_possible_actions(dummy_config, game_state.get_walls())

        # Update Configuration
        speed = 1.0

        possible_successors = []
        for action in possible_actions:
            vector = Actions.direction_to_vector(action, speed)
            successor = dummy_config.generate_successor(vector)
            possible_successors.append(successor.pos)

        return possible_successors, possible_actions


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAStarAgent', second='DefensiveReflexAgent', num_training=0):
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

# dijkstra's algorithm
def dik(agent_pos, goal, game_state=None):
    return 0
    
def aStarSearch(agent, goal, game_state, heuristic=dik):
    """Search the node that has the lowest combined cost and heuristic first."""

    # Dictionary to store each state's parent state and the action taken to reach it
    parents = {}

    # Set to keep track of visited states
    vis = set()

    # Dictionary to keep track of the nodes that are currently in the priority queue
    # and their associated costs (actual path cost and estimated total cost)
    in_queue = {}
    
    # Create a priority queue to manage states based on their estimated total cost
    q = util.PriorityQueue()
    
    # Get the starting state
    agent_pos = game_state.get_agent_position(agent.index)

    # Calculate the heuristic value for the starting state
    heuristic_val = heuristic(agent_pos, goal, game_state)

    # Push the starting state into the queue with its heuristic value and mark it as visited
    q.push(agent_pos, heuristic_val)
    vis.add(agent_pos)

    in_queue[agent_pos] = (0, heuristic_val)
    
    # Iterate over the queue until it is empty
    while not q.isEmpty():
        # Pop the state with the least estimated total cost from the priority queue
        cur_pos = q.pop()
        cur_path_cost, _ = in_queue.pop(cur_pos) # Retrieve and remove the current path cost from the in_queue dictionary
        
        # If this state is the goal state, set it as goal_state and break from the loop
        if goal == cur_pos:
            goal_state = cur_pos
            break
        
        
        legal_successors, legal_actions = get_theoretical_legal_successors(cur_pos, game_state)
         
        successors = []
        for successor, action in zip(legal_successors, legal_actions):
            successors.append((successor, action, 1))
                    
        # Iterate through the successors of the current state
        for pos, action, cost in successors:
            # Calculate the total path cost to reach the successor
            path_cost = cur_path_cost + cost

            # Calculate the estimated total cost for the successor (path cost + heuristic)
            total_cost = path_cost + heuristic(pos, goal, game_state)

            # If the successor is already in the priority queue and has a higher estimated total cost
            if pos in in_queue:
                if total_cost < in_queue[pos][1]:
                    # Update the estimated total cost in the priority queue and update the parent and action leading to the successor
                    q.update(pos, total_cost)
                    parents[pos] = (cur_pos, action)
            elif pos not in vis:
                # Mark the successor as visited
                vis.add(pos)

                # Store the successor's path cost and estimated total cost in the in_queue dictionary
                in_queue[pos] = (path_cost, total_cost)

                # Push the successor into the priority queue with its estimated total cost
                q.push(pos, total_cost)

                # Store the current state and action leading to the successor in the parents dictionary
                parents[pos] = (cur_pos, action)

    # Return the sequence of actions leading to the goal state
    return get_actions(goal_state, parents)

# Implements a FSM whose plan is determined by the game state \ heuristic space of the game state
# Abstract Class
class GoalPlanner:
    
    @staticmethod
    @abstractmethod
    def compute_goal(agent, game_state, current_plan, goal):
        pass


class GoalPlannerOffensive(GoalPlanner):
    @staticmethod
    def compute_goal(agent, game_state):
        
        # get the current position of the agent
        agent_pos = game_state.get_agent_position(agent.index)
        
        # get the food list
        if agent.red:
            food_list = game_state.get_blue_food().as_list()
        else:
            food_list = game_state.get_red_food().as_list()

        food_manhattan_distances_dict = {
            
        }
   
        for food_pos in food_list:
            food_manhattan_distances_dict[food_pos] = abs(food_pos[0] - agent_pos[0]) + abs(food_pos[1] - agent_pos[1])
        
        # sort the food list based on the manhattan distance
        food_manhattan_distances_dict_sorted = dict(sorted(food_manhattan_distances_dict.items(), key=lambda item: item[1]))
        
        
        # get the closest food position
        new_goal = list(food_manhattan_distances_dict_sorted.keys())[0]
        
        # if the goal is the same as the previous goal, then we don't need to recompute the plan
        if new_goal == agent.goal:
            return agent.goal
        else:
            return new_goal
        
        

# Updates the goal post dynamically based on the game state

class OffensiveAStarAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1, action_planner=GoalPlannerOffensive):
        super().__init__(index, time_for_computing)
        self.start = None
        self.goal = None
        self.plan = None
        self.action_planner = action_planner

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)
        CaptureAgent.register_initial_state(self, game_state)
        
    # Implements A* and executes the plan
    def choose_action(self, game_state):
        print(game_state)

        actions = game_state.get_legal_actions(self.index)
        
        if len(self.plan) <= 0:
            self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
            self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)

        if len(self.plan) > 0:
            next_action = self.plan.pop(0)
            if next_action in actions:
                return next_action
            else:
                self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
                self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)

            new_next_action = self.plan.pop(0)
            if new_next_action in actions:
                return new_next_action
            else:
                return random.choice(actions)
        else:
            return random.choice(actions)
        

# Based on the goal, finds the optimal path and executes it

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
