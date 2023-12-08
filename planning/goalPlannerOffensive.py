from planning.goalPlanner import GoalPlanner
from planning.goalPlannerDefensive import GoalPlannerDefensive
from planning.search import bfs_until_non_wall

class GoalPlannerOffensive(GoalPlanner):
    MAX_SAFE_DISTANCE = 8 # Max distance to an opponent ghost to be considered in safe mode
    BUFFER_ZONE_FROM_CENTER = 4 # Distance from the center of the board to consider the agent in a safe zone
    TIME_LIMIT_FACTOR = 2.5 # Factor to determine the time limit to return to the center
    SAFETY_MARGIN = 3 # Distance to maintain from the closest ghost
    FOOD_FRAC_TO_RETREAT = 2 # Fraction of food remaining to retreat
    MAX_OFFENSE_DISTANCE = 1 # Max distance to an opponent ghost to be considered in offensive mode
    POWER_PELLET_DISTANCE_MODIFIER = 2.5 # Modifier to determine if the agent should go for a power pellet instead of food
    POWER_PELLET_MIN_DISTANCE_FOR_EVAL = 3 # Min distance to a power pellet to be considered for evaluation against food

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

        # Ghost positions that will be useful for the next steps
        # Get opponent team members index and positions
        opponent_team_members = agent.get_opponents(game_state)
        opponent_ghosts_positions = {
            opponent: game_state.get_agent_position(opponent)
            for opponent in opponent_team_members if not game_state.get_agent_state(opponent).is_pacman
        }

        # Get the list of food
        food_list = game_state.get_blue_food().as_list() if agent.red else game_state.get_red_food().as_list()

        if food_list:
            closest_food_pos = min(food_list, key=lambda food: agent.get_maze_distance(agent_pos, food))
            distance_to_closest_food = agent.get_maze_distance(agent_pos, closest_food_pos)
        else:
            closest_food_pos = None
            distance_to_closest_food = float('inf')

        # Power pellet planning
        goal_for_power_pellet = GoalPlannerOffensive._plan_for_power_pellet(agent, game_state, agent_pos, opponent_ghosts_positions, opponent_team_members, distance_to_closest_food)
        if goal_for_power_pellet:
            return goal_for_power_pellet

        # Plan for avoiding ghosts
        goal_for_avoiding_ghosts = GoalPlannerOffensive._plan_for_avoiding_ghosts(agent, game_state, agent_is_pacman, x_distance_to_center, agent_pos, opponent_ghosts_positions)
        if goal_for_avoiding_ghosts:
            return goal_for_avoiding_ghosts

        # Default goal: go for food or return to center
        return GoalPlannerOffensive._default_goal(center_of_our_side, closest_food_pos, food_list, agent, agent_pos, game_state)

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
    def _check_goal_oscillation(agent, game_state, goal):
        previous_goals = agent.previous_goals

        # Check if the goal has been oscillating more than 4 times
        count_goal_oscillation = 0
        for i in range(len(previous_goals)-1):
            if previous_goals[i] == goal and previous_goals[i+1] != goal:
                count_goal_oscillation += 1
        if count_goal_oscillation > 3:
            return True
        else:
            return False

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
    def _plan_for_power_pellet(agent, game_state, agent_pos, opponent_ghosts_positions, opponent_team_members, distance_to_closest_food):
        
        # Get state of opponent ghosts
        opponent_ghosts_states = {
            opponent: game_state.get_agent_state(opponent)
            for opponent in opponent_team_members if not game_state.get_agent_state(opponent).is_pacman
        }

        all_ghosts_scared = all(ghost.scared_timer > 0 for ghost in opponent_ghosts_states.values())

        # Check if all opponent ghosts are scared, and if so, don't go for the power pellet
        if all_ghosts_scared:
            return None
        
        # Get the list of power pellets
        power_pellets = game_state.get_blue_capsules() if agent.red else game_state.get_red_capsules()
        if not power_pellets:
            return None

        # Find the closest power pellet
        closest_power_pellet = min(power_pellets, key=lambda pellet: agent.get_maze_distance(agent_pos, pellet))

        # Check if the power pellet position has been oscillating as the goal for the agent more than 4 times (to avoid oscillation)
        # It needs to count the non-consecutive appeareances of the power pellet as the goal with just one goal between each two appeareances
        power_pellet_oscillation = GoalPlannerOffensive._check_goal_oscillation(agent, game_state, closest_power_pellet)
        if power_pellet_oscillation:
            # If the power pellet position has been oscillating as the goal for the agent more than 4 times, don't go for the power pellet and
            # go for the second power pellet if available, and if not available, go for the closest food
            power_pellets_new = power_pellets.copy()
            power_pellets_new.remove(closest_power_pellet)
            if power_pellets_new:
                closest_power_pellet = min(power_pellets_new, key=lambda pellet: agent.get_maze_distance(agent_pos, pellet))

                # Check if the new power pellet position has been oscillating as the goal for the agent more than 4 times (to avoid oscillation)
                power_pellet_oscillation = GoalPlannerOffensive._check_goal_oscillation(agent, game_state, closest_power_pellet)
                if power_pellet_oscillation:
                    return None
            else:
                return None
            
        agent_distance_to_pellet = agent.get_maze_distance(agent_pos, closest_power_pellet)

        # If the distance to the closest power pellet is greater than a modifier of the distance to the closest food, don't go for the power pellet
        if agent_distance_to_pellet > GoalPlannerOffensive.POWER_PELLET_MIN_DISTANCE_FOR_EVAL:
            if agent_distance_to_pellet > distance_to_closest_food * GoalPlannerOffensive.POWER_PELLET_DISTANCE_MODIFIER:
                return None
        
        # Check if any ghost is closer to the power pellet than the agent
        for ghost_pos in opponent_ghosts_positions.values():
            if ghost_pos and agent.get_maze_distance(ghost_pos, closest_power_pellet) < agent_distance_to_pellet:
                return None  # Another ghost is closer, abort going for the pellet

        # Return the closest power pellet as the new goal
        return closest_power_pellet
    
    @staticmethod
    def _plan_for_avoiding_ghosts(agent, game_state, agent_is_pacman, x_distance_to_center, agent_pos, opponent_ghosts_positions):
        # Constants
        MAX_SAFE_DISTANCE = GoalPlannerOffensive.MAX_SAFE_DISTANCE
        BUFFER_ZONE_FROM_CENTER = GoalPlannerOffensive.BUFFER_ZONE_FROM_CENTER

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
    def _default_goal(center_of_our_side, closest_food_pos, food_list, agent, agent_pos, game_state):
        # Check if closest_food_pos is oscillating as the goal for the agent more than 4 times (to avoid oscillation)
        food_oscillation = GoalPlannerOffensive._check_goal_oscillation(agent, game_state, closest_food_pos)
        if food_oscillation:
            # Find the second closest food
            food_list_new = food_list.copy()
            food_list_new.remove(closest_food_pos)
            if food_list_new:
                closest_food_pos = min(food_list_new, key=lambda food: agent.get_maze_distance(agent_pos, food))

                # Check if the new food position has been oscillating as the goal for the agent more than 4 times (to avoid oscillation)
                food_oscillation = GoalPlannerOffensive._check_goal_oscillation(agent, game_state, closest_food_pos)
                if food_oscillation:
                    # Find the third closest food
                    food_list_new = food_list_new.copy()
                    food_list_new.remove(closest_food_pos)
                    if food_list_new:
                        closest_food_pos = min(food_list_new, key=lambda food: agent.get_maze_distance(agent_pos, food))

                        # Check if the new food position has been oscillating as the goal for the agent more than 4 times (to avoid oscillation)
                        food_oscillation = GoalPlannerOffensive._check_goal_oscillation(agent, game_state, closest_food_pos)
                        if food_oscillation:
                            # If the food position has been oscillating as the goal for the agent more than 4 times, go back to the center
                            return center_of_our_side
                        else:
                            return closest_food_pos
                    else:
                        # If there is no third closest food, go back to the center
                        return center_of_our_side
            else:
                # If there is no second closest food, go back to the center
                return center_of_our_side
            
        if closest_food_pos:
            return closest_food_pos
        else:
            # If no food is left, or as a fallback, return to the center of our side
            return center_of_our_side
