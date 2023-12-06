import random
import time

from enemy_localization.particleFilterAgent import ParticleFilterAgent
from planning.search import aStarSearch
from planning.util import is_legal_position, bfs_until_non_wall
from planning.goalPlannerDefensive import GoalPlannerDefensive

class DefensiveAStarAgent(ParticleFilterAgent):
    def __init__(
        self, index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector, time_for_computing=0.1, action_planner=GoalPlannerDefensive
    ):
        super().__init__(index, enemy_position_particle_filters, own_food_supervisor, enemy_suicide_detector, time_for_computing)
        self.goal = None
        self.plan = None
        self.smart_offensive_mode = False
        self.smart_offensive_timer = 0
        self.has_smart_defensive_offensive_capabilities = True
        self.defensive_roaming_goal = None
        self.action_planner = action_planner
        self.opponents_indexes = None
        self.enemy_position_estimates = None
        self.enemy_distance_estimates = None
        self.enemy_probabilistic_distances_estimates = None
        self.enemy_probabilistic_positions_estimates = None

        ## HYPERPARAMETERS ##
        self.OPPONENT_GHOST_WEIGHT = 20  # Cost for approaching an opponent ghost
        self.OPPONENT_GHOST_WEIGHT_ATTENUATION = 0.7 # Attenuation factor for the cost based on distance to the opponent ghost
        self.OPPONENT_PACMAN_WEIGHT = 8  # Reward for approaching an opponent pacman
        self.OPPONENT_PACMAN_WEIGHT_ATTENUATION = 0.3  # Attenuation factor for the reward based on distance to the opponent pacman
        self.ALLY_GHOST_WEIGHT = 8  # Cost for approaching an ally ghost
        self.ALLY_GHOST_WEIGHT_ATTENUATION = 0.3 # Attenuation factor for the cost based on distance to the ally ghost
        self.ALLY_PACMAN_WEIGHT = 4  # Cost for approaching an ally pacman
        self.ALLY_PACMAN_WEIGHT_ATTENUATION = 2  # Attenuation factor for the cost based on distance to the ally pacman

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.goal = self.action_planner.compute_goal(agent=self, game_state=game_state)
        # Fix goal if it is not legal
        self._fix_goal_if_not_legal(game_state)
        try:
            self.plan = aStarSearch(agent=self, goal=self.goal, game_state=game_state)
        except:
            self.plan = ['Stop', 'Stop', 'Stop', 'Stop', 'Stop', 'Stop', 'Stop']
        self.opponents_indexes = self.get_opponents(game_state)

    # Implements A* and executes the plan
    def choose_action(self, game_state):
        self.update_particle_filter(game_state)
        enemy_position_estimates = self.get_distinct_enemy_position_estimates()
        enemy_distance_estimates = self.get_distinct_enemy_distance_estimates()
        enemy_probabilistic_positions_estimates = self.get_probabilistic_enemy_position_estimates()
        enemy_probabilistic_distances_estimates = self.get_probabilistic_enemy_distance_estimates()

        # For the previous variables for the estimates, hold them in a dictionary with the enemy index as the key, knowing that the enemy index is not the same as those in the estimates, which follow the order of the opponents_indexes
        enemy_position_estimates_dict = {self.opponents_indexes[i]: enemy_position_estimates[i] for i in range(len(self.opponents_indexes))}
        enemy_distance_estimates_dict = {self.opponents_indexes[i]: enemy_distance_estimates[i] for i in range(len(self.opponents_indexes))}
        enemy_probabilistic_positions_estimates_dict = {self.opponents_indexes[i]: enemy_probabilistic_positions_estimates[i] for i in range(len(self.opponents_indexes))}
        enemy_probabilistic_distances_estimates_dict = {self.opponents_indexes[i]: enemy_probabilistic_distances_estimates[i] for i in range(len(self.opponents_indexes))}

        self.enemy_position_estimates = enemy_position_estimates_dict
        self.enemy_distance_estimates = enemy_distance_estimates_dict
        self.enemy_probabilistic_positions_estimates = enemy_probabilistic_positions_estimates_dict
        self.enemy_probabilistic_distances_estimates = enemy_probabilistic_distances_estimates_dict
        
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

    def defensive_heuristic(self, agent, goal, game_state, pos, track_heuristics=True, enable_profiling=False):
        heuristic = 0
        profiling_dict = {}
        heuristic_effect_dict = {}

        if enable_profiling:
            start_time = time.perf_counter()

        agent_is_pacman = game_state.get_agent_state(agent.index).is_pacman
        opponent_team_members = agent.opponents_indexes

        # Get the ally team members (the agent's team members that are not the agent)
        ally_team_members = [member for member in agent.get_team(game_state) if member != agent.index]

        # Categorize allies and adjust heuristic
        ally_ghost_distances, ally_pacman_distances = self._categorize_allies(game_state, ally_team_members, pos, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_allies(agent_is_pacman, ally_ghost_distances, ally_pacman_distances, heuristic_effect_dict if track_heuristics else None)

        # Categorize opponents and adjust heuristic
        opponent_ghost_distances, opponent_pacman_distances = self._categorize_opponents(game_state, opponent_team_members, pos, profiling_dict if enable_profiling else None)
        heuristic += self._adjust_heuristic_for_opponents(agent_is_pacman, opponent_ghost_distances, opponent_pacman_distances, heuristic_effect_dict if track_heuristics else None)

        # Strongly penalize the heuristic if the agent is not on smart offensive mode and the position being evaluated is in the
        # enemy's side of the board
        if not self.smart_offensive_mode:
            if self._is_in_enemy_side_of_board(agent, game_state, pos):
                heuristic -= 1000
                if heuristic_effect_dict is not None:
                    heuristic_effect_dict['enemy_side_of_board'] = heuristic

        if enable_profiling:
            profiling_dict['total_time'] = time.perf_counter() - start_time

        return heuristic
    
    # Helper function to check if the agent is in the enemy's side of the board
    def _is_in_enemy_side_of_board(self, agent, game_state, pos):
        # Get the center of the board
        center_of_board = (
            int(game_state.data.layout.width / 2),
            int(game_state.data.layout.height / 2),
        )

        # If agent is red, check if its x coordinate is more than the center of the board
        if agent.red and pos[0] > center_of_board[0]:
            return True
        # If agent is blue, check if its x coordinate is less than the center of the board
        elif not agent.red and pos[0] < center_of_board[0]:
            return True
        else:
            return False

        
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

            if is_legal_position(self.goal, game_state) == False:
                self.goal = bfs_until_non_wall(
                    self.goal, game_state
                )[-1]

    def _categorize_allies(self, game_state, ally_team_members, pos, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        ally_ghost_distances = {}
        ally_pacman_distances = {}
        for member in ally_team_members:
            distance = self.get_maze_distance(pos, game_state.get_agent_position(member))
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
    def _categorize_opponents(self, game_state, opponent_team_members, pos, profiling_dict=None):
        start_time = time.perf_counter() if profiling_dict is not None else None

        opponent_ghost_distances = {}
        opponent_pacman_distances = {}
        # If estimates are available, use the non-probabilistic estimates
        if self.enemy_position_estimates is not None :
            for member in opponent_team_members:
                distance = self.get_maze_distance(pos, self.enemy_position_estimates[member])
                if self.enemy_position_estimates[member] is not None:
                    if game_state.get_agent_state(member).is_pacman:
                        opponent_pacman_distances[member] = distance
                    else:
                        opponent_ghost_distances[member] = distance
        else:
            for member in opponent_team_members:
                distance = self.get_maze_distance(pos, game_state.get_agent_position(member))
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

