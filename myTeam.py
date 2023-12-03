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
from collections import deque
import random
import contest.util as util
import time

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
from queue import PriorityQueue
from contest.game import Directions, Configuration, Actions
from contest.capture import AgentRules


# This is required so our own files can be imported when the contest is run
import os
import sys

cd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cd)

from particleFilter import EnemyPositionParticleFilter
from ownFoodSupervisor import OwnFoodSupervisor
from enemySuicideDetector import EnemySuicideDetector
from particleFilterAgent import ParticleFilterAgent



#################
# Utility #
#################
def get_theoretical_legal_successors(position, game_state):
    """
    Returns the legal actions for an agent
    """
    dummy_config = Configuration(position, "North")
    possible_actions = Actions.get_possible_actions(
        dummy_config, game_state.get_walls()
    )

    # Update Configuration
    speed = 1.0

    possible_successors = []
    for action in possible_actions:
        vector = Actions.direction_to_vector(action, speed)
        successor = dummy_config.generate_successor(vector)
        possible_successors.append(successor.pos)

    return possible_successors, possible_actions


def bfs_until_non_wall(start, game_state):
    """
    Perform a breadth-first search until a non-wall position is found.

    :param start: Tuple (x, y) representing the start coordinate.
    :return: List of tuples representing the path to the first non-wall position.
    """

    # Correct start if its coordinates are negative
    if start[0] < 0:
        start = (0, start[1])
    if start[1] < 0:
        start = (start[0], 0)

    # Correct start if its coordinates are greater than the width or height of the board
    if start[0] >= game_state.data.layout.width:
        start = (game_state.data.layout.width - 1, start[1])
    if start[1] >= game_state.data.layout.height:
        start = (start[0], game_state.data.layout.height - 1)
        
    # Define movements: right, left, up, down
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Queue for BFS, initialized with the start position
    queue = deque([start])

    # Dictionary to keep track of visited nodes and their parents
    visited = {start: None}

    # Breadth-First Search
    while queue:
        current = queue.popleft()
        
        # check if current is in a valid position
        if current[1] >= game_state.data.layout.height or current[0] >= game_state.data.layout.width:
            continue

        # Check if current is not negative in any coordinate
        if current[0] < 0 or current[1] < 0:
            continue

        # Stop if the current position is not a wall
        if not game_state.has_wall(current[0], current[1]):
            path = []
            while current:
                path.append(current)
                current = visited[current]
            return path[::-1]  # Return reversed path

        # Check each neighbor
        for dx, dy in movements:
            neighbor = (current[0] + dx, current[1] + dy)

            # If the neighbor is not visited, add it to the queue and mark as visited
            if neighbor not in visited:
                queue.append(neighbor)
                visited[neighbor] = current

    # If a non-wall position is not reachable
    return None

# From a list of game states (observation history), extract the first definite position of any enemy agent
def get_first_definite_position(observation_history,agent, current_timeleft, time_limit):
    
    # Iterate through the observation history
    for game_state in reversed(observation_history):
        if game_state.data.timeleft - current_timeleft < time_limit:
            # Iterate through the opponent team
            for opponent in agent.get_opponents(game_state):
                # Get the position of the opponent
                opponent_pos = game_state.get_agent_position(opponent)

                # If the position is not None, return it
                if opponent_pos and game_state.get_agent_state(opponent).is_pacman == False and game_state.get_agent_state(opponent).scared_timer <= 0:
                    return opponent_pos

    # If no definite position was found, return None
    return None

# Get adjacent positions up to a certain distance, by adding and subtracting the distance to the x and y coordinates from (1 to distance)
def get_adjacent_positions(position, distance):
    adjacent_positions = []
    for i in range(1, distance + 1):
        adjacent_positions.append((position[0] + i, position[1]))
        adjacent_positions.append((position[0] - i, position[1]))
        adjacent_positions.append((position[0], position[1] + i))
        adjacent_positions.append((position[0], position[1] - i))
    return adjacent_positions


# Check if position is legal (not a wall) and not negative and not out of bounds based on board width and height
def is_legal_position(position, game_state):
    return (
        position[0] >= 0
        and position[1] >= 0
        and position[0] < game_state.data.layout.width
        and position[1] < game_state.data.layout.height
        and not game_state.has_wall(position[0], position[1])
    )



#################
# Team creation #
#################


def create_team(
    first_index,
    second_index,
    is_red,
    first="OffensiveSwitchAStarAgent",
    second="DefensiveAStarAgent",
    num_training=0,
):
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
    # the following objects are shared by both agents
    enemy_position_particle_filters = dict()
    own_food_supervisor = OwnFoodSupervisor()
    enemy_suicide_detector = EnemySuicideDetector()
    return [    
                OffensiveSwitchAStarAgent(first_index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector), 
                DefensiveAStarAgent(second_index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector)
            ]


##########
# Agents #
##########


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
def dijk(agent, goal, game_state=None):
    return 0


def aStarSearch(agent, goal, game_state, heuristic=dijk):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    WEIGHT = 1

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
    # time_heuristic = time.perf_counter()
    heuristic_val = heuristic(agent, goal, game_state)
    # print("Heuristic time: ", time.perf_counter() - time_heuristic)

    # Push the starting state into the queue with its heuristic value and mark it as visited
    q.push(agent_pos, heuristic_val)
    vis.add(agent_pos)

    in_queue[agent_pos] = (0, heuristic_val)

    # Iterate over the queue until it is empty
    while not q.isEmpty():
        # Pop the state with the least estimated total cost from the priority queue
        cur_pos = q.pop()
        cur_path_cost, _ = in_queue.pop(
            cur_pos
        )  # Retrieve and remove the current path cost from the in_queue dictionary

        # If this state is the goal state, set it as goal_state and break from the loop
        if goal == cur_pos:
            goal_state = cur_pos
            break

        legal_successors, legal_actions = get_theoretical_legal_successors(
            cur_pos, game_state
        )

        successors = []
        for successor, action in zip(legal_successors, legal_actions):
            successors.append((successor, action, 1))

        # Iterate through the successors of the current state
        for pos, action, cost in successors:
            # Calculate the total path cost to reach the successor
            path_cost = cur_path_cost + cost

            # Calculate the estimated total cost for the successor (path cost + heuristic)
            # time_heuristic = time.perf_counter()
            total_cost =  WEIGHT * path_cost + heuristic(agent, goal, game_state)
            # print("Heuristic time: ", time.perf_counter() - time_heuristic)

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
    MAX_SAFE_DISTANCE = 8 # Max distance to an opponent ghost to be considered in safe mode
    BUFFER_ZONE_FROM_CENTER = 4 # Distance from the center of the board to consider the agent in a safe zone
    TIME_LIMIT_FACTOR = 2 # Factor to determine the time limit to return to the center
    SAFETY_MARGIN = 3 # Distance to maintain from the closest ghost
    FOOD_FRAC_TO_RETREAT = 2 # Fraction of food remaining to retreat
    MAX_OFFENSE_DISTANCE = 2 # Max distance to an opponent ghost to be considered in offensive mode

    @staticmethod
    def compute_goal(agent, game_state):
        center_of_our_side = GoalPlannerOffensive._calculate_center_of_our_side(agent, game_state)
        agent_pos = game_state.get_agent_position(agent.index)
        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman
        x_distance_to_center = abs(agent_pos[0] - game_state.data.layout.width // 2)
    
        # Check if time is close to running out and we are winning so the agent becomes defensive
        if GoalPlannerOffensive._is_time_up_become_defensive(agent, game_state):
            agent.action_planner = GoalPlannerDefensive
            return GoalPlannerOffensive._defensive_roaming_mode(agent, game_state)
        
        # Check if it's time to retreat based on the game timer or if if pacman is carrying sufficient food to return to the center
        if GoalPlannerOffensive._is_time_to_retreat(agent, game_state):
            return GoalPlannerOffensive._determine_retreat_goal(agent, game_state, agent_is_pacman, center_of_our_side, agent_pos)

        # Power pellet planning
        goal_for_power_pellet = GoalPlannerOffensive._plan_for_power_pellet(agent, game_state, agent_pos)
        if goal_for_power_pellet:
            return goal_for_power_pellet

        # Plan for avoiding ghosts
        goal_for_avoiding_ghosts = GoalPlannerOffensive._plan_for_avoiding_ghosts(agent, game_state, agent_is_pacman, x_distance_to_center, agent_pos)
        if goal_for_avoiding_ghosts:
            return goal_for_avoiding_ghosts

        # Default goal: go for food or return to center
        return GoalPlannerOffensive._default_goal(agent, game_state, agent_pos, center_of_our_side)

    @staticmethod
    def _calculate_center_of_our_side(agent, game_state):
        # Calculate the center of our side
        width_factor = 1/4 if agent.red else 3/4
        height_factor = 1/4 if agent.red else 3/4

        center_of_our_side = (
            int(width_factor * game_state.data.layout.width),
            int(height_factor * game_state.data.layout.height)
        )

        # Adjust if there's a wall at the calculated position
        if game_state.has_wall(*center_of_our_side):
            center_of_our_side = bfs_until_non_wall(center_of_our_side, game_state)[-1]

        return center_of_our_side
    
    # Helper that after a certain time limit, the agent will become a defensive agent, if we are winning
    @staticmethod
    def _is_time_up_become_defensive(agent, game_state):

        # Get current time left and calculate the time limit
        time_left = game_state.data.timeleft
        limit_time_to_back_to_center = GoalPlannerOffensive.TIME_LIMIT_FACTOR * (game_state.data.layout.height + game_state.data.layout.width)
        time_up = time_left < limit_time_to_back_to_center

        # Check if we are winning
        score = game_state.get_score()
        if agent.red == False:
            score = -score

        return time_up and score > 0
    
    @staticmethod
    def _is_time_to_retreat(agent, game_state):
        time_left = game_state.data.timeleft
        limit_time_to_back_to_center = GoalPlannerOffensive.TIME_LIMIT_FACTOR * (game_state.data.layout.height + game_state.data.layout.width)
        time_up = time_left < limit_time_to_back_to_center
        
        # Check if pacman is carrying sufficient food to return to the center
        # sufficient is defined as a fraction of the food remaining
        food_remaining = len(agent.get_food(game_state).as_list())
        num_carrying = game_state.get_agent_state(agent.index).num_carrying
        food_sufficient = num_carrying >= food_remaining / GoalPlannerOffensive.FOOD_FRAC_TO_RETREAT
        
        return time_up or food_sufficient
    
    @staticmethod
    def _determine_retreat_goal(agent, game_state, agent_is_pacman, center_of_our_side, agent_pos):
        if agent_is_pacman:
            # Calculate a retreat point closer to our side of the board
            center_to_finish_game = (center_of_our_side[0], agent_pos[1])
            if game_state.has_wall(center_to_finish_game[0], center_to_finish_game[1]):
                center_to_finish_game = bfs_until_non_wall(center_to_finish_game, game_state)[-1]
            return center_to_finish_game
        else:
            # For non-pacman agents, go tinto defensive roaming mode
            return GoalPlannerOffensive._defensive_roaming_mode(agent, game_state)
        
    @staticmethod
    def _defensive_roaming_mode(agent, game_state):
        # First, check for nearby opponents
        closest_opponent, closest_opponent_distance = GoalPlannerOffensive._find_closest_opponent(agent, game_state)

        # If an opponent is close enough, and on our side of the board, go for it
        if closest_opponent and closest_opponent_distance < GoalPlannerOffensive.MAX_OFFENSE_DISTANCE and closest_opponent[0] < game_state.data.layout.width // 2- GoalPlannerOffensive.MAX_OFFENSE_DISTANCE:
            return closest_opponent
        
        
        # Otherwise, patrol areas with a lot of your team's food
        return GoalPlannerOffensive._patrol_food_rich_areas(agent, game_state)

    @staticmethod
    def _find_closest_opponent(agent, game_state):
        min_distance = float('inf')
        closest_opponent_position = None

        for opponent in agent.get_opponents(game_state):
            opponent_position = game_state.get_agent_position(opponent)
            if opponent_position:
                distance = agent.get_maze_distance(game_state.get_agent_position(agent.index), opponent_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_opponent_position = opponent_position

        return closest_opponent_position, min_distance

    @staticmethod
    def _patrol_food_rich_areas(agent, game_state):
        # Define the areas where your team's food is concentrated
        food_list = game_state.get_red_food().as_list() if agent.red else game_state.get_blue_food().as_list()

        # Find the area with the highest concentration of food
        food_centroid = GoalPlannerOffensive._calculate_food_centroid(game_state, food_list)
        
        # if agent is already very close the food centroid move away from it
        agent_pos = game_state.get_agent_position(agent.index)
        if agent_pos and food_centroid and agent.get_maze_distance(agent_pos, food_centroid) < 2:
            # go back to the center
            return GoalPlannerOffensive._calculate_center_of_our_side(agent, game_state)

        return food_centroid

    @staticmethod
    def _calculate_food_centroid(game_state,food_list):
        if not food_list:
            return None

        x_sum, y_sum = 0, 0
        for food in food_list:
            x_sum += food[0]
            y_sum += food[1]
            
        centroid = (int(x_sum / len(food_list)), int(y_sum / len(food_list)))
        
        # wall check
        if game_state.has_wall(centroid[0], centroid[1]):
            centroid = bfs_until_non_wall(centroid, game_state)[-1]

        return centroid

    
    @staticmethod
    def _plan_for_power_pellet(agent, game_state, agent_pos):
        # Get the list of power pellets
        power_pellets = game_state.get_blue_capsules() if agent.red else game_state.get_red_capsules()
        if not power_pellets:
            return None

        # Find the closest power pellet
        closest_power_pellet = min(power_pellets, key=lambda pellet: agent.get_maze_distance(agent_pos, pellet))
        agent_distance_to_pellet = agent.get_maze_distance(agent_pos, closest_power_pellet)

        # Get opponent ghosts' positions
        opponent_team_members = agent.get_opponents(game_state)
        opponent_ghosts_positions = [game_state.get_agent_position(opponent) for opponent in opponent_team_members if not game_state.get_agent_state(opponent).is_pacman]

        # Check if any ghost is closer to the power pellet than the agent
        for ghost_pos in opponent_ghosts_positions:
            if ghost_pos and agent.get_maze_distance(ghost_pos, closest_power_pellet) < agent_distance_to_pellet:
                return None  # Another ghost is closer, abort going for the pellet

        # Return the closest power pellet as the new goal
        return closest_power_pellet
    
    @staticmethod
    def _plan_for_avoiding_ghosts(agent, game_state, agent_is_pacman, x_distance_to_center, agent_pos):
        # Constants
        MAX_SAFE_DISTANCE = GoalPlannerOffensive.MAX_SAFE_DISTANCE
        BUFFER_ZONE_FROM_CENTER = GoalPlannerOffensive.BUFFER_ZONE_FROM_CENTER

        # Get opponent team members index and positions
        opponent_team_members = agent.get_opponents(game_state)
        opponent_ghosts_positions = {
            opponent: game_state.get_agent_position(opponent)
            for opponent in opponent_team_members if not game_state.get_agent_state(opponent).is_pacman
        }

        # Filter to get only the opponent ghosts that are not scared
        opponent_ghosts_positions = {opp: pos for opp, pos in opponent_ghosts_positions.items() if pos and game_state.get_agent_state(opp).scared_timer == 0}

        # Find the closest opponent ghost
        closest_ghost_distance, closest_ghost_position = None, None
        for ghost, pos in opponent_ghosts_positions.items():
            distance = agent.get_maze_distance(agent_pos, pos)
            if closest_ghost_distance is None or distance < closest_ghost_distance:
                closest_ghost_distance, closest_ghost_position = distance, pos

        # Determine if the agent needs to avoid the closest ghost
        if closest_ghost_distance and closest_ghost_distance <= MAX_SAFE_DISTANCE:
            if agent_is_pacman or (not agent_is_pacman and x_distance_to_center <= BUFFER_ZONE_FROM_CENTER):
                # Calculate safe position to retreat
                # This can be a predefined safe location or dynamically calculated
                return GoalPlannerOffensive._calculate_safe_retreat(agent, game_state, closest_ghost_position)

        return None
    
    @staticmethod
    def _calculate_safe_retreat(agent, game_state, closest_ghost_position):
        # Constants for calculations
        SAFETY_MARGIN = GoalPlannerOffensive.SAFETY_MARGIN  # distance to maintain from the closest ghost

        agent_pos = game_state.get_agent_position(agent.index)

        # Calculate retreat positions towards power pellets, teammates, and home area
        retreat_options = []
        
        # 1. Consider moving towards power pellets if available
        power_pellets = game_state.get_blue_capsules() if agent.red else game_state.get_red_capsules()
        for pellet in power_pellets:
            retreat_options.append((pellet, 'power_pellet'))

        # 2. Consider moving towards teammates
        teammates = agent.get_team(game_state)
        teammates.remove(agent.index)  # Exclude the current agent
        for teammate in teammates:
            teammate_pos = game_state.get_agent_position(teammate)
            retreat_options.append((teammate_pos, 'teammate'))

        # 3. Consider moving towards the home area
        home_x, _ = GoalPlannerOffensive._calculate_center_of_our_side(agent, game_state)
        home_area_positions = [(home_x, y) for y in range(game_state.data.layout.height)]
        for pos in home_area_positions:
            if not game_state.has_wall(*pos):
                retreat_options.append((pos, 'home_area'))

        # Evaluate the best retreat option based on distance and safety
        best_retreat, best_score = None, float('inf')
        for pos, _ in retreat_options:
            distance = agent.get_maze_distance(agent_pos, pos)
            safety_distance = agent.get_maze_distance(pos, closest_ghost_position)
            score = distance - safety_distance * SAFETY_MARGIN  # Prioritize safety over proximity

            if score < best_score:
                best_retreat, best_score = pos, score

        if best_retreat:
            return best_retreat

        # Fallback: Directly away from the ghost if no other option is viable
        vector_away_from_ghost = (agent_pos[0] - closest_ghost_position[0], agent_pos[1] - closest_ghost_position[1])
        magnitude = max(abs(vector_away_from_ghost[0]), abs(vector_away_from_ghost[1]))
        if magnitude != 0:
            direction_away_from_ghost = (vector_away_from_ghost[0] / magnitude, vector_away_from_ghost[1] / magnitude)
        else:
            direction_away_from_ghost = (0, 0)

        potential_retreat_pos = (
            int(agent_pos[0] + direction_away_from_ghost[0] * SAFETY_MARGIN),
            int(agent_pos[1] + direction_away_from_ghost[1] * SAFETY_MARGIN),
        )

        return GoalPlannerOffensive._adjust_retreat_position(potential_retreat_pos, game_state)

    
    @staticmethod
    def _adjust_retreat_position(potential_retreat_pos, game_state):
        # Ensure the indices are within the game layout bounds
        max_x, max_y = game_state.data.layout.width - 1, game_state.data.layout.height - 1
        x, y = min(max(0, potential_retreat_pos[0]), max_x), min(max(0, potential_retreat_pos[1]), max_y)

        if game_state.has_wall(x, y):
            # Find the nearest non-wall position
            return bfs_until_non_wall((x, y), game_state)[-1]
        return (x, y)

    
    @staticmethod
    def _default_goal(agent, game_state, agent_pos, center_of_our_side):
        # Check if there's any food left to eat
        food_list = game_state.get_blue_food().as_list() if agent.red else game_state.get_red_food().as_list()

        if food_list:
            # Find the closest food position
            closest_food_pos = min(food_list, key=lambda food: agent.get_maze_distance(agent_pos, food))
            return closest_food_pos
        else:
            # If no food is left, or as a fallback, return to the center of our side
            return center_of_our_side


class OffensiveSwitchAStarAgent(ParticleFilterAgent):
    def __init__(
        self, index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector, time_for_computing=0.1, action_planner=GoalPlannerOffensive
    ):
        super().__init__(index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector, time_for_computing)
        ## HYPERPARAMETERS ##
        self.OPPONENT_GHOST_WEIGHT = 5  # Cost for approaching an opponent ghost
        self.OPPONENT_GHOST_WEIGHT_ATTENUATION = 0.5 # Attenuation factor for the cost based on distance to the opponent ghost
        self.OPPONENT_PACMAN_WEIGHT = 7  # Reward for approaching an opponent pacman
        self.OPPONENT_PACMAN_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the opponent pacman
        self.POWER_PELLET_WEIGHT = 0.5  # Reward for approaching a power pellet
        self.POWER_PELLET_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the power pellet
        self.SCARED_GHOST_REWARD = 8  # Reward for approaching a scared ghost
        self.SCARED_GHOST_DISTANCE_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the scared ghost
        self.GHOST_COLLISION_PENALTY = 20  # Penalty for states closer to a previously known ghost location
        self.GHOST_COLLISION_DISTANCE_ATTENUATION = 0.2  # Attenuation factor for the penalty based on distance to the previously known ghost location
        self.EPSILON = 0.001  # Small value to avoid division by zero
        self.goal = None
        self.plan = None
        self.action_planner = action_planner
        self.has_smart_defensive_offensive_capabilities = False
        self.defensive_roaming_goal = None


    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)

    # Implements A* and executes the plan
    def choose_action(self, game_state):
        self.update_particle_filter(game_state)
        enemy_position_estimates = self.get_enemy_position_estimates()
        enemy_distance_estimates = self.get_enemy_distance_estimates()

        # start_goal_calc = time.perf_counter()
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        # print("Goal calc time: ", time.perf_counter() - start_goal_calc)
        # start_plan_calc = time.perf_counter()

        # Fix goal if it is not legal
        self._fix_goal_if_not_legal(game_state)

        self.plan = aStarSearch(
            agent=self,
            goal=self.goal,
            game_state=game_state,
            heuristic=self.offensive_heuristic,
        )
        # print("Plan calc time: ", time.perf_counter() - start_plan_calc)

        actions = game_state.get_legal_actions(self.index)

        if len(self.plan) > 0:
            next_action = self.plan.pop(0)
            if next_action in actions:
                return next_action
            else:
                self.goal = self.action_planner.compute_goal(
                    agent=self, game_state=game_state
                )
                self.plan = aStarSearch(
                    agent=self, goal=self.goal, game_state=game_state
                )

            new_next_action = self.plan.pop(0)
            if new_next_action in actions:
                return new_next_action
            else:
                return random.choice(actions)
        else:
            return random.choice(actions)
        
    def offensive_heuristic(self, agent, goal, game_state, track_heuristics=False, enable_profiling=False):
        heuristic = 0
        profiling_dict = {}
        heuristic_effect_dict = {}

        if enable_profiling:
            start_time = time.perf_counter()

        agent_pos = game_state.get_agent_position(agent.index)
        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman
        opponent_team_members = agent.get_opponents(game_state)

        # Categorize opponents and adjust heuristic
        opponent_ghost_distances, opponent_pacman_distances = self._categorize_opponents(game_state, opponent_team_members, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_opponents(agent_is_pacman, opponent_ghost_distances, opponent_pacman_distances, heuristic_effect_dict if track_heuristics else None)

        # Power pellets
        power_pellet_list = self._get_power_pellet_list(agent, game_state, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_power_pellets(agent_pos, power_pellet_list, heuristic_effect_dict if track_heuristics else None)

        # Scared ghosts
        scared_ghosts = self._get_scared_ghosts(opponent_team_members, game_state, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_scared_ghosts(agent_pos, scared_ghosts, heuristic_effect_dict if track_heuristics else None)

        # Ghost collisions
        if not agent_is_pacman:
            heuristic += self._adjust_heuristic_for_ghost_collisions(agent_pos, game_state, profiling_dict if enable_profiling else None)

        if enable_profiling:
            profiling_dict['total_time'] = time.perf_counter() - start_time

        return heuristic
    
    # Helper function that fixes the goal if it is not legal
    def _fix_goal_if_not_legal(self, game_state):
        # If any of the coordinates of the goal is negative, set that coordinate to 0
        if self.goal[0] < 0:
            self.goal = (0, self.goal[1])
        if self.goal[1] < 0:
            self.goal = (self.goal[0], 0)

        # If any of the coordinates of the goal is greater than the width or height of the board, set that coordinate to the width or height of the board
        if self.goal[0] >= game_state.data.layout.width:
            self.goal = (game_state.data.layout.width - 1, self.goal[1])
        if self.goal[1] >= game_state.data.layout.height:
            self.goal = (self.goal[0], game_state.data.layout.height - 1)

        if is_legal_position(self.goal, game_state) == False:
            self.goal = (
                int(game_state.data.layout.width / 2),
                int(game_state.data.layout.height / 2),
            )

    # Helper function to categorize opponents into ghosts and pacmen
    def _categorize_opponents(self, game_state, opponent_team_members, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        opponent_ghost_distances = {}
        opponent_pacman_distances = {}
        for member in opponent_team_members:
            distance = game_state.agent_distances[member]
            if game_state.get_agent_state(member).is_pacman:
                opponent_pacman_distances[member] = distance
            else:
                opponent_ghost_distances[member] = distance

        if profiling_dict is not None:
            profiling_dict['_categorize_opponents'] = time.perf_counter() - start_time

        return opponent_ghost_distances, opponent_pacman_distances
    
    # Helper function to adjust the heuristic based on the opponents
    def _adjust_heuristic_for_opponents(self, agent_is_pacman, opponent_ghost_distances, opponent_pacman_distances, heuristic_effect_dict=None):
        heuristic = 0
        if agent_is_pacman and opponent_ghost_distances:
            closest_ghost_distance = min(opponent_ghost_distances.values())
            heuristic += self.OPPONENT_GHOST_WEIGHT / (max(closest_ghost_distance, 1) ** self.OPPONENT_GHOST_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['opponent_ghost'] = heuristic

        if not agent_is_pacman and opponent_pacman_distances:
            closest_pacman_distance = min(opponent_pacman_distances.values())
            heuristic -= self.OPPONENT_PACMAN_WEIGHT / (max(closest_pacman_distance, 1) ** self.OPPONENT_PACMAN_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['opponent_pacman'] = heuristic

        return heuristic
    
    # Helper function to get the power pellet list
    def _get_power_pellet_list(self, agent, game_state, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        # Get the list of power pellets based on the agent's team
        if agent.red:
            power_pellet_list = game_state.get_blue_capsules()
        else:
            power_pellet_list = game_state.get_red_capsules()

        if profiling_dict is not None:
            profiling_dict['_get_power_pellet_list'] = time.perf_counter() - start_time

        return power_pellet_list
    
    # Helper function to adjust the heuristic based on the power pellets
    def _adjust_heuristic_for_power_pellets(self, agent_pos, power_pellet_list, heuristic_effect_dict=None):
        heuristic = 0
        if power_pellet_list:
            # Calculate the minimum distance to a power pellet
            min_power_pellet_distance = min(
                [self.get_maze_distance(agent_pos, power_pellet) for power_pellet in power_pellet_list]
            )

            # Adjust the heuristic based on the distance to the nearest power pellet
            heuristic -= self.POWER_PELLET_WEIGHT / (min_power_pellet_distance ** self.POWER_PELLET_WEIGHT_ATTENUATION + self.EPSILON)

            if heuristic_effect_dict is not None:
                heuristic_effect_dict['power_pellet'] = heuristic

        return heuristic
    
    # Helper function to get the scared ghosts
    def _get_scared_ghosts(self, opponent_team_members, game_state, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        scared_ghosts = {}
        for opponent in opponent_team_members:
            opponent_state = game_state.get_agent_state(opponent)
            if opponent_state.scared_timer > 1:
                scared_ghosts[opponent] = game_state.get_agent_position(opponent)

        if profiling_dict is not None:
            profiling_dict['_get_scared_ghosts'] = time.perf_counter() - start_time

        return scared_ghosts
    
    # Helper function to adjust the heuristic based on the scared ghosts
    def _adjust_heuristic_for_scared_ghosts(self, agent_pos, scared_ghosts, heuristic_effect_dict=None):
        heuristic = 0
        for scared_ghost_pos in scared_ghosts.values():
            if scared_ghost_pos:
                distance_to_scared_ghost = self.get_maze_distance(agent_pos, scared_ghost_pos)
                heuristic -= self.SCARED_GHOST_REWARD / (distance_to_scared_ghost ** self.SCARED_GHOST_DISTANCE_ATTENUATION + self.EPSILON)

                if heuristic_effect_dict is not None:
                    heuristic_effect_dict['scared_ghost'] = heuristic

        return heuristic
    
    # Helper function to adjust the heuristic based on the ghost collisions
    def _adjust_heuristic_for_ghost_collisions(self, agent_pos, game_state, profiling_dict=None):
        heuristic = 0

        if profiling_dict is not None:
            start_time = time.perf_counter()

        # Assume last_enemy_pos is a method or attribute that gives the last known position of the nearest enemy ghost
        last_enemy_pos = last_enemy_pos = get_first_definite_position( self.observationHistory, self, game_state.data.timeleft, 75)


        if last_enemy_pos:
            distance_to_last_known_ghost = self.get_maze_distance(agent_pos, last_enemy_pos)
            # The closer the agent is to the last known position of a ghost, the higher the penalty
            heuristic += self.GHOST_COLLISION_PENALTY / (distance_to_last_known_ghost ** self.GHOST_COLLISION_DISTANCE_ATTENUATION + self.EPSILON)

        if profiling_dict is not None:
            profiling_dict['_adjust_heuristic_for_ghost_collisions'] = time.perf_counter() - start_time

        return heuristic


# DEFENSIVE
class GoalPlannerDefensive(GoalPlanner):
    SAFE_DISTANCE = 5
    ROAM_LIMIT_MAX = 5
    ROAM_LIMIT_MIN = 2
    PERCENTAGE_FOOD_PELLETS_SMART_OFFENSIVE = 0.1
    LIMIT_TIMER_SMART_OFFENSIVE = 50
    LIMIT_SMART_OFFENSIVE_CLOSE_FOOD = 6
    SMART_OFFENSIVE_CLOSE_FOOD_MULTIPLIER = 2.5
    GET_AWAY_FROM_ALLY_GHOSTS_DISTANCE = 6
    
    @staticmethod
    def compute_goal(agent, game_state):
        # Get useful information
        agent_pos = game_state.get_agent_position(agent.index)
        agent_state = game_state.get_agent_state(agent.index)
        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman

        # Get the center of the board
        center_of_board = (
            int(game_state.data.layout.width / 2),
            int(game_state.data.layout.height / 2),
        )

        # Enemy Pacman Positions
        enemy_pacman_positions = {
            opponent: game_state.get_agent_position(opponent)
            for opponent in agent.get_opponents(game_state)
            if game_state.get_agent_state(opponent).is_pacman
        }
        # Remove None values
        enemy_pacman_positions = {
            k: v for k, v in enemy_pacman_positions.items() if v is not None
        }

        # Distance to center
        x_distance_to_center = abs(agent_pos[0] - game_state.data.layout.width // 2)

        # Enemy Ghost Positions
        enemy_ghost_positions = {
            opponent: game_state.get_agent_position(opponent)
            for opponent in agent.get_opponents(game_state)
            if not game_state.get_agent_state(opponent).is_pacman and game_state.get_agent_state(opponent).scared_timer == 0
        }

        # Remove None values
        enemy_ghost_positions = {
            k: v for k, v in enemy_ghost_positions.items() if v is not None
        }

        # Ally Ghost Positions
        ally_ghost_positions = {
            teammate: game_state.get_agent_position(teammate)
            for teammate in agent.get_team(game_state)
            if not game_state.get_agent_state(teammate).is_pacman and game_state.get_agent_state(teammate).scared_timer == 0
        }

        
        # Smart offensive only if agent has smart offensive capabilities
        if agent.has_smart_defensive_offensive_capabilities:
            # Smart Offensive Mode Goal: If the agent has eaten an enemy pacman and there's not another enemy pacman nearby and is close to the center of the board (vertical line), if there are any
            # close food pellets in the enemy side of the board, go for them up to a limit of 10% of the total food pellets in the enemy side of the board
            smart_offensive_goal = GoalPlannerDefensive._has_eaten_enemy_pacman_and_no_other_close_enemies(game_state, agent, agent_is_pacman, agent_pos, agent_state, enemy_ghost_positions, x_distance_to_center)
            if smart_offensive_goal:
                return smart_offensive_goal
                    
        
        # Evade Invader Goal: If the Agent is Scared, is a Ghost and can see an invader, set the goal to evade the invader by a safe distance margin
        evade_invader_goal = GoalPlannerDefensive._evade_invader_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos, game_state, center_of_board)
        if evade_invader_goal:
            return evade_invader_goal
         
        # Closest Invader Targeting Goal: If the agent can see an invader and is a ghost, set the goal to chase the invader
        closest_invader_targeting_goal = GoalPlannerDefensive._closest_invader_targeting_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos)
        if closest_invader_targeting_goal:
            return closest_invader_targeting_goal

        # Target Recently Eaten Food: If the agent is a ghost and can see recently eaten food, set the goal to go to the closest food pellet that has been eaten by the opponent team
        recently_eaten_food_goal = GoalPlannerDefensive._recently_eaten_food_goal(agent, game_state, agent_pos)
        if recently_eaten_food_goal:
            return recently_eaten_food_goal
    
        
        # Default goal: Roaming Logic to get close to the center of the board in order to block enemy advances and get ready for smart offensives,
        # trying to get away from ally ghosts
        return GoalPlannerDefensive._default_goal(agent, game_state, center_of_board, agent_pos, ally_ghost_positions)
    

    # Smart Offensive Mode: If the agent has eaten an enemy pacman and there's not another enemy pacman nearby and is close to the center of the board (vertical line), if there are any
    # close food pellets in the enemy side of the board, go for them up to a limit of 10% of the total food pellets in the enemy side of the board
    @staticmethod
    def _has_eaten_enemy_pacman_and_no_other_close_enemies(game_state, agent, agent_is_pacman, agent_pos, agent_state, opponent_ghosts_positions, x_distance_to_center):
        if agent._has_eaten_enemy_pacman_and_no_other_close_enemies(game_state) and agent._is_close_to_center(game_state, GoalPlannerDefensive.LIMIT_SMART_OFFENSIVE_CLOSE_FOOD):
            agent.smart_offensive_mode = True
            agent.smart_offensive_timer = 0
        
        if agent.smart_offensive_timer > GoalPlannerDefensive.LIMIT_TIMER_SMART_OFFENSIVE:
            agent.smart_offensive_mode = False
            agent.smart_offensive_timer = 0


        if agent.smart_offensive_mode:
            # Retreat logic from GoalPlannerOffensive
            # Constants
            MAX_SAFE_DISTANCE = GoalPlannerOffensive.MAX_SAFE_DISTANCE
            BUFFER_ZONE_FROM_CENTER = GoalPlannerOffensive.BUFFER_ZONE_FROM_CENTER

            # Find the closest opponent ghost
            closest_ghost_distance, closest_ghost_position = None, None
            for ghost, pos in opponent_ghosts_positions.items():
                distance = agent.get_maze_distance(agent_pos, pos)
                if closest_ghost_distance is None or distance < closest_ghost_distance:
                    closest_ghost_distance, closest_ghost_position = distance, pos

            # Determine if the agent needs to avoid the closest ghost
            if closest_ghost_distance and closest_ghost_distance <= MAX_SAFE_DISTANCE:
                if agent_is_pacman or (not agent_is_pacman and x_distance_to_center <= BUFFER_ZONE_FROM_CENTER):
                    # Calculate safe position to retreat
                    # This can be a predefined safe location or dynamically calculated
                    return GoalPlannerOffensive._calculate_safe_retreat(agent, game_state, closest_ghost_position)

            enemy_food = agent.get_food(game_state).as_list()
            enemy_food_limit = int(len(enemy_food) * GoalPlannerDefensive.PERCENTAGE_FOOD_PELLETS_SMART_OFFENSIVE)  # 10% of enemy food pellets

            if agent_state.num_carrying >= enemy_food_limit:
                agent.smart_offensive_mode = False
                agent.smart_offensive_timer = 0
            else:
                # Find food pellets close to the center line on the enemy side
                close_enemy_food = [
                    food for food in enemy_food 
                    if abs(food[0] - game_state.data.layout.width // 2) <= GoalPlannerDefensive.LIMIT_SMART_OFFENSIVE_CLOSE_FOOD
                    and agent.get_maze_distance(agent_pos, food) <= GoalPlannerDefensive.LIMIT_SMART_OFFENSIVE_CLOSE_FOOD * GoalPlannerDefensive.SMART_OFFENSIVE_CLOSE_FOOD_MULTIPLIER
                ]

                # If there are close food pellets on the enemy side, choose the closest one
                if close_enemy_food:
                    closest_food = min(
                        close_enemy_food, 
                        key=lambda food: agent.get_maze_distance(agent_pos, food)
                    )
                    return closest_food
                
        else:
            # If agent is a pacman and not on smart offensive, set the goal to return to our side of the board
            if agent_is_pacman:
                if agent.red:
                    return bfs_until_non_wall((
                        int(game_state.data.layout.width / 4),
                        agent_pos[1]), game_state
                    )[-1]
                else:
                    return bfs_until_non_wall((
                        int(3 * game_state.data.layout.width / 4),
                        agent_pos[1]), game_state)[-1]


    # Evade Invader Goal: If the Agent is Scared, is a Ghost and can see an invader, set the goal to evade the invader by a safe distance margin
    @staticmethod
    def _evade_invader_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos, game_state, center_of_board):
        # If the Agent is Scared, is a Ghost and can see an invader, set the goal to evade the invader by a safe distance margin
        if agent_is_pacman == False and game_state.get_agent_state(agent.index).scared_timer > 0 and len(enemy_pacman_positions) > 0:
            closest_invader = min(
                enemy_pacman_positions,
                key=lambda opponent: agent.get_maze_distance(
                    agent_pos, enemy_pacman_positions[opponent]
                ),
            )
            closest_invader_pos = enemy_pacman_positions[closest_invader]
            # If the closest invader is close enough, set the goal to evade the invader calculating the closest safe position that is not a wall
            # and is closer to the center of the board, but not crossing the center of the board (so when it resets, we can defend again)
            if agent.get_maze_distance(agent_pos, closest_invader_pos) <= GoalPlannerDefensive.SAFE_DISTANCE:
                # If agent is red, the closest safe position is to the right of the enemy as long as it does not cross the center of the board
                # If it crosses, go up or down if possible
                if agent.red:
                    closest_safe_position = (
                        closest_invader_pos[0] + GoalPlannerDefensive.SAFE_DISTANCE,
                        closest_invader_pos[1],
                    )
                    if closest_safe_position[0] > center_of_board[0]:
                        closest_safe_position = (
                            closest_invader_pos[0],
                            closest_invader_pos[1] + GoalPlannerDefensive.SAFE_DISTANCE,
                        )

                        # Check if the closest safe position is a wall or is valid by checking the height of the board and if its negative
                        if is_legal_position(closest_safe_position, game_state) == False:
                            closest_safe_position = (
                                closest_invader_pos[0],
                                closest_invader_pos[1] - GoalPlannerDefensive.SAFE_DISTANCE,
                            )
                        
                else:
                    closest_safe_position = (
                        closest_invader_pos[0] - GoalPlannerDefensive.SAFE_DISTANCE,
                        closest_invader_pos[1],
                    )
                    if closest_safe_position[0] < center_of_board[0]:
                        closest_safe_position = (
                            closest_invader_pos[0],
                            closest_invader_pos[1] + GoalPlannerDefensive.SAFE_DISTANCE,
                        )

                        # Check if the closest safe position is a wall or is valid by checking the height of the board and if its negative
                        if is_legal_position(closest_safe_position, game_state) == False:
                            closest_safe_position = (
                                closest_invader_pos[0],
                                closest_invader_pos[1] - GoalPlannerDefensive.SAFE_DISTANCE,
                            )
                        

                return bfs_until_non_wall(closest_safe_position, game_state)[-1]
            
    # Closest Invader Targeting Goal: If the agent can see an invader and is a ghost, set the goal to chase the invader
    @staticmethod
    def _closest_invader_targeting_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos):
        if agent_is_pacman == False and len(enemy_pacman_positions) > 0:
            closest_invader = min(
                enemy_pacman_positions,
                key=lambda opponent: agent.get_maze_distance(
                    agent_pos, enemy_pacman_positions[opponent]
                ),
            )
            return enemy_pacman_positions[closest_invader]


    # Recently Eaten Food Goal: If the agent is a ghost and can see recently eaten food, set the goal to go to the closest food pellet that has been eaten by the opponent team
    @staticmethod
    def _recently_eaten_food_goal(agent, game_state, agent_pos):
        food_list = agent.get_food_you_are_defending(game_state).as_list()
        for i in range(2, min(15, len(agent.observationHistory))):
            # Get the food list of the previous game state according to the observation history
            previous_food_list = agent.get_food_you_are_defending(
                agent.observationHistory[-i]
            ).as_list()

            # Get the food pellets that have been eaten by the opponent team
            eaten_food_list = list(set(previous_food_list) - set(food_list))

            # If there are food pellets that have been eaten by the opponent team, set the goal to the closest food pellet that has not been eaten
            # by the opponent team and is in the agent's side
            if len(eaten_food_list) > 0:
                closest_food = min(
                    eaten_food_list,
                    key=lambda food: agent.get_maze_distance(agent_pos, food),
                )
                return closest_food
            

    # Default goal: Roaming Logic: If the agent is not scared, cannot see any invaders and is a ghost, set the goal to roam around key areas which
    # are defined as being close to the border up to a distance between 2 and 5 of the center line of the board.
    # Determine the border range based on the agent's side (red or blue)
    @staticmethod
    def _default_goal(agent, game_state, center_of_board, agent_pos, ally_ghost_positions):
        if agent.defensive_roaming_goal:
            if agent_pos == agent.defensive_roaming_goal:
                agent.defensive_roaming_goal = None

            if agent.defensive_roaming_goal:
                return agent.defensive_roaming_goal
        
        if agent.red:
            border_x = range(center_of_board[0] - GoalPlannerDefensive.ROAM_LIMIT_MAX, center_of_board[0] - GoalPlannerDefensive.ROAM_LIMIT_MIN)
        else:
            border_x = range(center_of_board[0] + GoalPlannerDefensive.ROAM_LIMIT_MIN, center_of_board[0] + GoalPlannerDefensive.ROAM_LIMIT_MAX)

        # Generate a list of potential goal positions near the border
        potential_goals = []
        for x in border_x:
            for y in range(1, game_state.data.layout.height - 1):  # Avoid the very top and bottom
                if not game_state.has_wall(x, y):
                    potential_goals.append((x, y))

        non_close_to_allies_potential_goals = [goal for goal in potential_goals]
        # Remove positions from the list of potential goals that are too close to ally ghosts
        for ally_ghost_pos in ally_ghost_positions.values():
            if ally_ghost_pos:
                non_close_to_allies_potential_goals = [
                    goal
                    for goal in non_close_to_allies_potential_goals
                    if agent.get_maze_distance(goal, ally_ghost_pos) > GoalPlannerDefensive.GET_AWAY_FROM_ALLY_GHOSTS_DISTANCE
                ]

        # If there are potential goals that are not too close to ally ghosts, choose a random one and set a roaming goal
        if non_close_to_allies_potential_goals:
            agent.defensive_roaming_goal = random.choice(non_close_to_allies_potential_goals)
            return agent.defensive_roaming_goal

        # Choose a random goal from the list of potential goals if the previous check fails
        if potential_goals:
            return random.choice(potential_goals)
        
        return game_state.data.layout.getRandomLegalPosition()
    


class DefensiveAStarAgent(CaptureAgent):
    def __init__(
        self, index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector, time_for_computing=0.1, action_planner=GoalPlannerDefensive
    ):
        super().__init__(index, time_for_computing)
        self.start = None
        self.goal = None
        self.plan = None
        self.smart_offensive_mode = False
        self.smart_offensive_timer = 0
        self.has_smart_defensive_offensive_capabilities = True
        self.defensive_roaming_goal = None
        self.action_planner = action_planner

        ## HYPERPARAMETERS ##
        self.OPPONENT_GHOST_WEIGHT = 5  # Cost for approaching an opponent ghost
        self.OPPONENT_GHOST_WEIGHT_ATTENUATION = 0.5 # Attenuation factor for the cost based on distance to the opponent ghost
        self.OPPONENT_PACMAN_WEIGHT = 7  # Reward for approaching an opponent pacman
        self.OPPONENT_PACMAN_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the opponent pacman
        self.ALLY_GHOST_WEIGHT = 5  # Cost for approaching an ally ghost
        self.ALLY_GHOST_WEIGHT_ATTENUATION = 0.5 # Attenuation factor for the cost based on distance to the ally ghost
        self.ALLY_PACMAN_WEIGHT = 7  # Cost for approaching an ally pacman
        self.ALLY_PACMAN_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the cost based on distance to the ally pacman

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)

    # Implements A* and executes the plan
    def choose_action(self, game_state):
        if self.smart_offensive_mode:
            self.smart_offensive_timer += 1

        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)

        # Fix goal if it is not legal
        self._fix_goal_if_not_legal(game_state)

        self.plan = aStarSearch(
            agent=self,
            goal=self.goal,
            game_state=game_state,
            heuristic=self.defensive_heuristic,
        )

        actions = game_state.get_legal_actions(self.index)

        if len(self.plan) > 0:
            next_action = self.plan.pop(0)
            if next_action in actions:
                return next_action
            else:
                self.goal = self.action_planner.compute_goal(
                    agent=self, game_state=game_state
                )
                self.plan = aStarSearch(
                    agent=self, goal=self.goal, game_state=game_state
                )

            new_next_action = self.plan.pop(0)
            if new_next_action in actions:
                return new_next_action
            else:
                return random.choice(actions)
        else:
            return random.choice(actions)

    def defensive_heuristic(self, agent, goal, game_state, track_heuristics=True, enable_profiling=False):
        heuristic = 0
        profiling_dict = {}
        heuristic_effect_dict = {}

        if enable_profiling:
            start_time = time.perf_counter()

        agent_pos = game_state.get_agent_position(agent.index)
        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman
        opponent_team_members = agent.get_opponents(game_state)

        # Categorize allies and adjust heuristic
        ally_ghost_distances, ally_pacman_distances = self._categorize_allies(game_state, opponent_team_members, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_allies(agent_is_pacman, ally_ghost_distances, ally_pacman_distances, heuristic_effect_dict if track_heuristics else None)

        # Categorize opponents and adjust heuristic
        opponent_ghost_distances, opponent_pacman_distances = self._categorize_opponents(game_state, opponent_team_members, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_opponents(agent_is_pacman, opponent_ghost_distances, opponent_pacman_distances, heuristic_effect_dict if track_heuristics else None)

        if enable_profiling:
            profiling_dict['total_time'] = time.perf_counter() - start_time

        return heuristic
    
    # Helper function that fixes the goal if it is not legal
    def _fix_goal_if_not_legal(self, game_state):
        # If any of the coordinates of the goal is negative, set that coordinate to 0
        if self.goal[0] < 0:
            self.goal = (0, self.goal[1])
        if self.goal[1] < 0:
            self.goal = (self.goal[0], 0)

        # If any of the coordinates of the goal is greater than the width or height of the board, set that coordinate to the width or height of the board
        if self.goal[0] > game_state.data.layout.width:
            self.goal = (game_state.data.layout.width, self.goal[1])
        if self.goal[1] > game_state.data.layout.height:
            self.goal = (self.goal[0], game_state.data.layout.height)

        if is_legal_position(self.goal, game_state) == False:
            self.goal = (
                int(game_state.data.layout.width / 2),
                int(game_state.data.layout.height / 2),
            )

    def _categorize_allies(self, game_state, ally_team_members, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        ally_ghost_distances = {}
        ally_pacman_distances = {}
        for member in ally_team_members:
            distance = game_state.agent_distances[member]
            if game_state.get_agent_state(member).is_pacman:
                ally_pacman_distances[member] = distance
            else:
                ally_ghost_distances[member] = distance

        if profiling_dict is not None:
            profiling_dict['_categorize_allies'] = time.perf_counter() - start_time

        return ally_ghost_distances, ally_pacman_distances
    
    # Helper function to check if the agent has eaten an enemy pacman and there are no other close enemies
    def _has_eaten_enemy_pacman_and_no_other_close_enemies(self, game_state):
        # Get the previous game state if it exists
        if len(self.observationHistory) > 1:
            previous_game_state = self.observationHistory[-2]

            # Get the opponent team members index
            opponent_team_members = self.get_opponents(game_state)

            # Get the opponent team members positions in the previous game state and the current game state. If before the opponent team member
            # was not None, but now it is, and the distance from the agent now to the opponent team member in the previous game state is less than
            # 2, then the agent has eaten an enemy pacman
            previous_opponent_team_members_pos = {
                opponent: previous_game_state.get_agent_position(opponent)
                for opponent in opponent_team_members
            }
            current_opponent_team_members_pos = {
                opponent: game_state.get_agent_position(opponent)
                for opponent in opponent_team_members
            }
            
            # Distance from the agent now to the opponent team member in the previous game state is less than 2
            distance_condition = lambda opponent: self.get_maze_distance(
                game_state.get_agent_position(self.index),
                previous_opponent_team_members_pos[opponent],
            ) <= 2

            # If before the opponent team member was not None, but now it is, and the distance from the agent now to the opponent team member
            # in the previous game state is less than 2, then the agent has eaten an enemy pacman
            eaten_condition = lambda opponent: (
                previous_opponent_team_members_pos[opponent]
                and current_opponent_team_members_pos[opponent] is None
                and distance_condition(opponent)
            )

            # Check if there is another enemy pacman nearby after eating the enemy pacman by checking if any of the current opponent team members
            # positions is not None and the distance from the agent to the opponent team member is less than 5
            current_opponent_condition = lambda opponent: (
                current_opponent_team_members_pos[opponent]
                and self.get_maze_distance(
                    game_state.get_agent_position(self.index),
                    current_opponent_team_members_pos[opponent],
                )
                <= 5
            )
                

            # If there is at least one opponent team member that has been eaten, return True
            if any(eaten_condition(opponent) for opponent in opponent_team_members) and not any(current_opponent_condition(opponent) for opponent in opponent_team_members):
                return True
            else:
                return False
        else:
            return False
        
    # Helper function to check if the agent is close to the center of the board
    def _is_close_to_center(self, game_state, distance=5):
        # Get the center of the board
        center_of_board = (
            int(game_state.data.layout.width / 2),
            int(game_state.data.layout.height / 2),
        )

        # Get the agent's position
        agent_pos = game_state.get_agent_position(self.index)

        # Calculate the distance to the center of the board
        distance_to_center = abs(agent_pos[0] - center_of_board[0])

        # If the distance to the center of the board is less than 5, return True
        if distance_to_center <= distance:
            return True
        else:
            return False
        
    # Helper function to categorize opponents into ghosts and pacmen
    def _categorize_opponents(self, game_state, opponent_team_members, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        opponent_ghost_distances = {}
        opponent_pacman_distances = {}
        for member in opponent_team_members:
            distance = game_state.agent_distances[member]
            if game_state.get_agent_state(member).is_pacman:
                opponent_pacman_distances[member] = distance
            else:
                opponent_ghost_distances[member] = distance

        if profiling_dict is not None:
            profiling_dict['_categorize_opponents'] = time.perf_counter() - start_time

        return opponent_ghost_distances, opponent_pacman_distances

    # Helper function to adjust the heuristic based on the opponents
    def _adjust_heuristic_for_opponents(self, agent_is_pacman, opponent_ghost_distances, opponent_pacman_distances, heuristic_effect_dict=None):
        heuristic = 0
        if agent_is_pacman and opponent_ghost_distances:
            closest_ghost_distance = min(opponent_ghost_distances.values())
            heuristic += self.OPPONENT_GHOST_WEIGHT / (max(closest_ghost_distance, 1) ** self.OPPONENT_GHOST_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['opponent_ghost'] = heuristic

        if not agent_is_pacman and opponent_pacman_distances:
            closest_pacman_distance = min(opponent_pacman_distances.values())
            heuristic -= self.OPPONENT_PACMAN_WEIGHT / (max(closest_pacman_distance, 1) ** self.OPPONENT_PACMAN_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['opponent_pacman'] = heuristic

        return heuristic
    
    # Helper function to adjust the heuristic based on the allies
    def _adjust_heuristic_for_allies(self, agent_is_pacman, ally_ghost_distances, ally_pacman_distances, heuristic_effect_dict=None):
        heuristic = 0
        if agent_is_pacman and ally_ghost_distances:
            closest_ghost_distance = min(ally_ghost_distances.values())
            heuristic += self.ALLY_GHOST_WEIGHT / (max(closest_ghost_distance, 1) ** self.ALLY_GHOST_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['ally_ghost'] = heuristic

        if not agent_is_pacman and ally_pacman_distances:
            closest_pacman_distance = min(ally_pacman_distances.values())
            heuristic += self.ALLY_PACMAN_WEIGHT / (max(closest_pacman_distance, 1) ** self.ALLY_PACMAN_WEIGHT_ATTENUATION)
            if heuristic_effect_dict is not None:
                heuristic_effect_dict['ally_pacman'] = heuristic

        return heuristic

