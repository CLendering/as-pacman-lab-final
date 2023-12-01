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
    # Define movements: right, left, up, down
    movements = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    # Queue for BFS, initialized with the start position
    queue = deque([start])

    # Dictionary to keep track of visited nodes and their parents
    visited = {start: None}

    # Breadth-First Search
    while queue:
        current = queue.popleft()

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

#################
# Team creation #
#################


def create_team(
    first_index,
    second_index,
    is_red,
    first="OffensiveAStarAgent",
    second="DefensiveReflexAgent",
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=0.1):
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
        features["successor_score"] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {"successor_score": 1.0}


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
    MAX_SAFE_DISTANCE = 8
    BUFFER_ZONE_FROM_CENTER = 4

    @staticmethod
    def compute_goal(agent, game_state):
        # Constants
        # Get the center of our side position in x, y coordinates
        if agent.red:
            center_of_our_side = (
                int(game_state.data.layout.width / 4),
                int(game_state.data.layout.height / 4),
            )
            if game_state.has_wall(center_of_our_side[0], center_of_our_side[1]):
                center_of_our_side = bfs_until_non_wall(center_of_our_side, game_state)[
                    -1
                ]
        else:
            center_of_our_side = (
                int(3 * game_state.data.layout.width / 4),
                int(3 * game_state.data.layout.height / 4),
            )
            if game_state.has_wall(center_of_our_side[0], center_of_our_side[1]):
                center_of_our_side = bfs_until_non_wall(center_of_our_side, game_state)[
                    -1
                ]
            
        # get the current position of the agent
        agent_pos = game_state.get_agent_position(agent.index)
        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman

        # Calculate the distance from the agent to the center vertical line (x distance)
        center_of_board = (
            int(game_state.data.layout.width / 2),
            int(game_state.data.layout.height / 2),
        )
        x_distance_to_center = abs(agent_pos[0] - center_of_board[0])

        # Get the remaining timer for the game
        time_left = game_state.data.timeleft
        
        # Function of Height and Width of the board
        limit_time_to_back_to_center =  2.5*(game_state.data.layout.height + game_state.data.layout.width)
        
        if time_left < limit_time_to_back_to_center:

            if agent_is_pacman:
                # We calculate the new center as being the same x but the y of the center of our side
                center_to_finish_game = (center_of_our_side[0], agent_pos[1])
                if game_state.has_wall(center_to_finish_game[0], center_to_finish_game[1]):
                    center_to_finish_game = bfs_until_non_wall(center_to_finish_game, game_state)[-1]
                return center_to_finish_game
 
            else:
                # Roaming Defensive Thing
                # @TODO
                pass


        # get the other team members index
        other_team_members = agent.get_team(game_state)
        other_team_members.remove(agent.index)

        # get the other team members positions as a dictionary with the key being the index and the value being the position
        other_team_members_pos = {}
        for other_team_member in other_team_members:
            other_team_members_pos[other_team_member] = game_state.get_agent_position(
                other_team_member
            )

        # get opponent team members index
        opponent_team_members = agent.get_opponents(game_state)

        # get opponent team members positions as a dictionary with the key being the index and the value being the position
        opponent_team_members_pos = {}
        for opponent_team_member in opponent_team_members:
            opponent_team_members_pos[
                opponent_team_member
            ] = game_state.get_agent_position(opponent_team_member)

        # Power pellet planning
        # Check if opponent team members pos are all None
        if not all(
            opponent_team_member_pos is None
            for opponent_team_member_pos in opponent_team_members_pos.values()
        ):
            # get the power pellet list
            power_pellets = (
                game_state.get_blue_capsules()
                if agent.red
                else game_state.get_red_capsules()
            )
            if len(power_pellets)>0:
                closest_power_pellet = min(
                    power_pellets,
                    key=lambda pellet: agent.get_maze_distance(agent_pos, pellet),
                )
                agent_distance_to_pellet = agent.get_maze_distance(
                    agent_pos, closest_power_pellet
                )

                # Assume ghosts will move towards the power pellet as well
                ghost_distances_to_pellet = [
                    agent.get_maze_distance(ghost_pos, closest_power_pellet)
                    for ghost_pos in opponent_team_members_pos.values()
                    if ghost_pos
                ]

                # If the agent can reach the power pellet before any ghost, then go for it
                if not ghost_distances_to_pellet or agent_distance_to_pellet < min(
                    ghost_distances_to_pellet
                ):
                    return closest_power_pellet
                

        # get the distance to the nearest opponent that is a ghost
        # Filter the list of opponents to only include ghosts
        opponent_ghosts = list(
            filter(
                lambda x: game_state.get_agent_state(x).is_pacman == False,
                opponent_team_members,
            )
        )
        position_and_distance_to_opponent_ghosts = {}
        for opponent_ghost in opponent_ghosts:
            if opponent_team_members_pos[opponent_ghost]:
                position_and_distance_to_opponent_ghosts[
                    opponent_ghost
                ] = agent.get_maze_distance(
                    agent_pos, opponent_team_members_pos[opponent_ghost]
                )

        if len(position_and_distance_to_opponent_ghosts) > 0:
            # Get the closest opponent ghost index
            closest_opponent_ghost_index = min(
                position_and_distance_to_opponent_ghosts,
                key=position_and_distance_to_opponent_ghosts.get,
            )

            # Get the closest opponent ghost position
            closest_opponent_ghost_pos = opponent_team_members_pos[
                closest_opponent_ghost_index
            ]

            # Check for the presence of scared ghosts before deciding to retreat
            scared_ghosts = {
                opponent: game_state.get_agent_state(opponent)
                for opponent in agent.get_opponents(game_state)
                if game_state.get_agent_state(opponent).scared_timer > 0
            }

            # If there are scared ghosts, do not retreat to center, instead continue with other objectives
            if not scared_ghosts:
                # if the agent is a pacman and the closest opponent ghost is within a certain distance, then we need to run away
                if (
                    agent_is_pacman
                    or (
                        agent_is_pacman == False
                        and x_distance_to_center
                        <= GoalPlannerOffensive.BUFFER_ZONE_FROM_CENTER
                    )
                ) and position_and_distance_to_opponent_ghosts[
                    closest_opponent_ghost_index
                ] <= GoalPlannerOffensive.MAX_SAFE_DISTANCE:
                    return center_of_our_side

        # get the food list
        if agent.red:
            food_list = game_state.get_blue_food().as_list()
        else:
            food_list = game_state.get_red_food().as_list()

        if len(food_list) > 0:
            maze_distance_dict = {}

            for food_pos in food_list:
                maze_distance_dict[food_pos] = agent.get_maze_distance(
                    agent_pos, food_pos
                )


            # sort the food list based on the manhattan distance
            maze_distance_dict_sorted = dict(
                sorted(maze_distance_dict.items(), key=lambda item: item[1])
            )

            # get the closest food position
            new_goal = list(maze_distance_dict_sorted.keys())[0]
        else:
            new_goal = center_of_our_side

        # if the goal is the same as the previous goal, then we don't need to recompute the plan
        if new_goal == agent.goal:
            return agent.goal
        else:
            return new_goal


# Updates the goal post dynamically based on the game state


class OffensiveAStarAgent(CaptureAgent):
    def __init__(
        self, index, time_for_computing=0.1, action_planner=GoalPlannerOffensive
    ):
        super().__init__(index, time_for_computing)
        ## PENALTY FOR STATES WITH GHOSTS NEARBY
        self.OPPONENT_GHOST_WEIGHT = 4  # Reward for approaching an opponent ghost
        self.OPPONENT_GHOST_WEIGHT_ATTENUATION = 0.5 # Attenuation factor for the reward based on distance to the opponent ghost
        self.OPPONENT_PACMAN_WEIGHT = 4  # Reward for approaching an opponent pacman
        self.OPPONENT_PACMAN_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the opponent pacman
        self.POWER_PELLET_WEIGHT = 0.5  # Reward for approaching a power pellet
        self.POWER_PELLET_WEIGHT_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the power pellet
        self.SCARED_GHOST_REWARD = 8  # Reward for approaching a scared ghost
        self.SCARED_GHOST_DISTANCE_ATTENUATION = 0.5  # Attenuation factor for the reward based on distance to the scared ghost
        self.GHOST_COLLISION_PENALTY = 20  # Penalty for states closer to a previously known ghost location
        self.GHOST_COLLISION_DISTANCE_ATTENUATION = 0.2  # Attenuation factor for the penalty based on distance to the previously known ghost location
        self.EPSILON = 0.001  # Small value to avoid division by zero
        self.start = None
        self.goal = None
        self.plan = None
        self.action_planner = action_planner

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)

    # Implements A* and executes the plan
    def choose_action(self, game_state):
        # start_goal_calc = time.perf_counter()
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        # print("Goal calc time: ", time.perf_counter() - start_goal_calc)
        # start_plan_calc = time.perf_counter()

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

    # Penalty for states with ghosts nearby
    # Bonus for power pellets along the path
    # Bonus for regions with a lot of food pellets
    def offensive_heuristic(self, agent, goal, game_state):

        heuristic = 0

        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman

        heuristic_effect_dict = {}

        # profiling_dict = {}

        # get the index of opponent team members
        # time_get_opponent_team_members = time.perf_counter()
        opponent_team_members = agent.get_opponents(game_state)

        # profiling_dict["time_get_opponent_team_members"] = time.perf_counter() - time_get_opponent_team_members

        # get the absolute value of the noisy estimate of ghost distance
        # time_get_noisy_ghost_distances = time.perf_counter()
        distances = game_state.agent_distances
        # profiling_dict["time_get_noisy_ghost_distances"] = time.perf_counter() - time_get_noisy_ghost_distances

        # opponent distances to agent
        opponent_ghost_distances = {}
        opponent_pacman_distances = {}

        # time_get_opponent_ghost_distances = time.perf_counter()
        for opponent_team_member in opponent_team_members:
            # if the opponent is a ghost
            if game_state.get_agent_state(opponent_team_member).is_pacman == False:
                opponent_ghost_distances[opponent_team_member] = distances[
                    opponent_team_member
                ]
            else:
                opponent_pacman_distances[opponent_team_member] = distances[
                    opponent_team_member
                ]
        # profiling_dict["time_get_opponent_ghost_distances"] = time.perf_counter() - time_get_opponent_ghost_distances

        # time_get_agent_pacman_and_sum_heuristic = time.perf_counter()
        if len(opponent_ghost_distances) > 0:
            # get the closest distance to an opponent
            closest_opponent_distance = min(opponent_ghost_distances.values())
            closest_opponent_distance = max(closest_opponent_distance, 0)
            if agent_is_pacman:
                heuristic += self.OPPONENT_GHOST_WEIGHT/ (closest_opponent_distance**self.OPPONENT_GHOST_WEIGHT_ATTENUATION + self.EPSILON)
                heuristic_effect_dict["opponent_ghost"] = self.OPPONENT_GHOST_WEIGHT/ (closest_opponent_distance**self.OPPONENT_GHOST_WEIGHT_ATTENUATION + self.EPSILON)
        if len(opponent_pacman_distances) > 0:
            # get the closest distance to a pacman
            closest_pacman_distance = min(opponent_pacman_distances.values())
            closest_pacman_distance = max(closest_pacman_distance, 0)

            # if the agent is a ghost
            if agent_is_pacman == False:
                heuristic -= self.OPPONENT_PACMAN_WEIGHT / (closest_pacman_distance**self.OPPONENT_PACMAN_WEIGHT_ATTENUATION + self.EPSILON)
                heuristic_effect_dict["opponent_pacman"] = self.OPPONENT_PACMAN_WEIGHT / (closest_pacman_distance**self.OPPONENT_PACMAN_WEIGHT_ATTENUATION + self.EPSILON)
        # profiling_dict["time_get_agent_pacman_and_sum_heuristic"] = time.perf_counter() - time_get_agent_pacman_and_sum_heuristic

        ## BONUS FOR POWER PELLETS ALONG THE PATH

        #time_power_pellet_list = time.perf_counter()
        # get the power pellet list
        if agent.red:
            power_pellet_list = game_state.get_blue_capsules()
        else:
            power_pellet_list = game_state.get_red_capsules()

        # compute the minimium distance to a power pellet
        if len(power_pellet_list) > 0:
            min_power_pellet_distance = min(
                [
                    agent.get_maze_distance(
                        game_state.get_agent_position(agent.index), power_pellet
                    )
                    for power_pellet in power_pellet_list
                ]
            )
            heuristic -= self.POWER_PELLET_WEIGHT/ (min_power_pellet_distance**self.POWER_PELLET_WEIGHT_ATTENUATION + self.EPSILON)
            heuristic_effect_dict["power_pellet"] = self.POWER_PELLET_WEIGHT/ (min_power_pellet_distance**self.POWER_PELLET_WEIGHT_ATTENUATION + self.EPSILON)
        # profiling_dict["time_power_pellet_list"] = time.perf_counter() - time_power_pellet_list

        agent_pos = game_state.get_agent_position(agent.index)

        ## BONUS FOR ATTACKING A SCARED GHOST
        # time_bonus_attack_scared_ghost = time.perf_counter()

        # Get scared ghosts and their positions
        scared_ghosts = {
            opponent: game_state.get_agent_state(opponent)
            for opponent in opponent_team_members
            if not game_state.get_agent_state(opponent).is_pacman
            and game_state.get_agent_state(opponent).scared_timer > 0
        }

        # Calculate the reward for approaching scared ghosts
        for scared_ghost_index, scared_ghost_state in scared_ghosts.items():
            scared_ghost_pos = game_state.get_agent_position(scared_ghost_index)
            # check if the scared ghost is not None
            if scared_ghost_pos:
                # Calculate the distance to the scared ghost
                distance_to_scared_ghost = agent.get_maze_distance(
                    agent_pos, scared_ghost_pos
                )
            else:
                distance_to_scared_ghost = 9999
            # Subtract from heuristic to reward being closer to the scared ghost
            heuristic -= self.SCARED_GHOST_REWARD / (
                distance_to_scared_ghost**self.SCARED_GHOST_DISTANCE_ATTENUATION + self.EPSILON
            )  # Add 1 to avoid division by zero
            heuristic_effect_dict["scared_ghost"] = self.SCARED_GHOST_REWARD / (
                distance_to_scared_ghost**self.SCARED_GHOST_DISTANCE_ATTENUATION + self.EPSILON
            )

        # profiling_dict["time_bonus_attack_scared_ghost"] = time.perf_counter() - time_bonus_attack_scared_ghost
        
        # Penalties for states closer to previously known ghost locations IF we are at our side of the board
        if agent_is_pacman == False:
            last_enemy_pos = get_first_definite_position(
                self.observationHistory, self, game_state.data.timeleft, 75
            )

            if last_enemy_pos:
                heuristic += self.GHOST_COLLISION_PENALTY / (agent.get_maze_distance(agent_pos, last_enemy_pos)**self.GHOST_COLLISION_DISTANCE_ATTENUATION + self.EPSILON)
                heuristic_effect_dict["ghost_collision"] = self.GHOST_COLLISION_PENALTY / (agent.get_maze_distance(agent_pos, last_enemy_pos)**self.GHOST_COLLISION_DISTANCE_ATTENUATION + self.EPSILON)
        return heuristic


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
        features["successor_score"] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if (
            len(food_list) > 0
        ):  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min(
                [self.get_maze_distance(my_pos, food) for food in food_list]
            )
            features["distance_to_food"] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {"successor_score": 100, "distance_to_food": -1}


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
        features["on_defense"] = 1
        if my_state.is_pacman:
            features["on_defense"] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features["num_invaders"] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features["invader_distance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1
        rev = Directions.REVERSE[
            game_state.get_agent_state(self.index).configuration.direction
        ]
        if action == rev:
            features["reverse"] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            "num_invaders": -1000,
            "on_defense": 100,
            "invader_distance": -10,
            "stop": -100,
            "reverse": -2,
        }
