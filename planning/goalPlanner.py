from abc import abstractmethod

# Implements a FSM whose plan is determined by the game state \ heuristic space of the game state
# Abstract Class
class GoalPlanner:
    @abstractmethod
    def compute_goal(agent, game_state, current_plan, goal):
        pass
