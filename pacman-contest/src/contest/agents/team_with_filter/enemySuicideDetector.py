from contest.util import manhattanDistance
from contest.capture import SIGHT_RANGE

class EnemySuicideDetector():
    """
    Checks whether an enemy died in their own turn.
    This is important to know because in that case, we have to reset
    their particle filter to their spawn position.
    """
    def __init__(self):
        self.initialized = False

    def initialize(self, enemies, agentsOnTeam, teamSpawnPositions):
        if self.initialized:
            print('EnemySuicideDetector.initialize was called although it was already initialized before')
        self.enemies = enemies
        self.agentsOnTeam = agentsOnTeam
        self.teamSpawnPositions = teamSpawnPositions
        self.enemiesToCloseAgentsOnTeam = {enemy: set() for enemy in enemies}
        self.__enemySuicides = {enemy: False for enemy in enemies}

        self.initialized = True

    def update(self, game_state):
        # reset suicides
        for enemy in self.enemies:
            if self.__enemySuicides[enemy]:
                self.__enemySuicides[enemy] = False

        friendly_agent_positions = [game_state.get_agent_position(friend) for friend in self.agentsOnTeam]

        for enemy in self.enemies:
            enemy_position = game_state.get_agent_position(enemy)
            # Enemy is visible
            if enemy_position is not None:
                # Remember which agents are close to the enemy
                for friend, position in zip(self.agentsOnTeam, friendly_agent_positions):
                    if manhattanDistance(enemy_position, position) <= SIGHT_RANGE // 2:
                        self.enemiesToCloseAgentsOnTeam[enemy].add(friend)
                    else:
                        # friend is close enough to see the enemy, but not close enough for the enemy to commit suicide
                        self.enemiesToCloseAgentsOnTeam[enemy].discard(friend)
            # Enemy is not visible
            else:
                # If enemy was close to a friendly agent before
                # and that friendly agent is still alive, 
                # the enemy committed suicide in his own turn
                for friend_close_to_enemy in self.enemiesToCloseAgentsOnTeam[enemy]:
                    if game_state.get_agent_position(friend_close_to_enemy) != self.teamSpawnPositions[friend_close_to_enemy]:
                        self.__enemySuicides[enemy] = True
                        break
                
                self.enemiesToCloseAgentsOnTeam[enemy] = set()

    
    def hasCommittedSuicide(self, enemy):
        return self.__enemySuicides[enemy]