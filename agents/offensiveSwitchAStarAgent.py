import random
import time

from enemy_localization.particleFilterAgent import ParticleFilterAgent
from planning.search import aStarSearch
from planning.util import is_legal_position, get_first_definite_position
from planning.goalPlannerOffensive import GoalPlannerOffensive

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
        # TODO use enemy position/distance estimates

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
