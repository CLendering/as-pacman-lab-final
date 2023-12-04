from planning.goalPlanner import GoalPlanner
from planning.search import bfs_until_non_wall
from planning.util import is_legal_position
import random

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

        # Enemy Pacman Positions: Use estimates for enemy pacman positions if the agent has them
        if agent.enemy_position_estimates:
            enemy_pacman_positions = {
                opponent: agent.enemy_position_estimates[opponent]
                for opponent in agent.get_opponents(game_state)
                if game_state.get_agent_state(opponent).is_pacman
            }
        else:
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
        closest_invader_targeting_goal = GoalPlannerDefensive._closest_invader_targeting_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos, center_of_board)
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
        # avoid circular import
        from planning.goalPlannerOffensive import GoalPlannerOffensive
        
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
            
    # Closest Invader Targeting Goal: If the agent can see an invader (or has an estimate) and is a ghost, set the goal to chase the invader
    @staticmethod
    def _closest_invader_targeting_mode(agent, agent_is_pacman, enemy_pacman_positions, agent_pos, center_of_board):
        if agent_is_pacman == False and len(enemy_pacman_positions) > 0:
            closest_invader = min(
                enemy_pacman_positions,
                key=lambda opponent: agent.get_maze_distance(
                    agent_pos, enemy_pacman_positions[opponent]
                ),
            )

            closest_invader_pos = enemy_pacman_positions[closest_invader]

            # If the closest invader position estimate is at the enemy side of the board, clamp the goal to be before the center of the board
            if agent.red:
                if closest_invader_pos[0] > center_of_board[0]:
                    closest_invader_pos = (
                        center_of_board[0] - 1,
                        closest_invader_pos[1],
                    )
            else:
                if closest_invader_pos[0] < center_of_board[0]:
                    closest_invader_pos = (
                        center_of_board[0] + 1,
                        closest_invader_pos[1],
                    )
            
            return closest_invader_pos


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
    
