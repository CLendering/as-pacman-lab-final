# captureGraphicsDisplay.py
# -------------------------
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


from contest.graphicsUtils import *
import math, time
from contest.game import Directions
from contest.util import Counter

###########################
#  GRAPHICS DISPLAY CODE  #
###########################

# Most code by Dan Klein and John Denero written or rewritten for cs188, UC Berkeley.
# Some code from a Pacman implementation by LiveWires, and used / modified with permission.

DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 35
BACKGROUND_COLOR = formatColor(0, 0, 0)
WALL_COLOR = formatColor(0.0 / 255.0, 51.0 / 255.0, 255.0 / 255.0)
INFO_PANE_COLOR = formatColor(.4, .4, 0)
SCORE_COLOR = formatColor(.9, .9, .9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = []
GHOST_COLORS.append(formatColor(.9, 0, 0))  # Red
GHOST_COLORS.append(formatColor(0, .3, .9))  # Blue
GHOST_COLORS.append(formatColor(.98, .41, .07))  # Orange
GHOST_COLORS.append(formatColor(.1, .75, .7))  # Green
GHOST_COLORS.append(formatColor(1.0, 0.6, 0.0))  # Yellow
GHOST_COLORS.append(formatColor(.4, 0.13, 0.91))  # Purple

TEAM_COLORS = GHOST_COLORS[:2]

GHOST_SHAPE = [
    (0, 0.3),
    (0.25, 0.75),
    (0.5, 0.3),
    (0.75, 0.75),
    (0.75, -0.5),
    (0.5, -0.75),
    (-0.5, -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5, 0.3),
    (-0.25, 0.75)
]
GHOST_SIZE = 0.65
SCARED_COLOR = formatColor(1, 1, 1)

# GHOST_VEC_COLORS = map(colorToVector, GHOST_COLORS)
GHOST_VEC_COLORS = [colorToVector(c) for c in GHOST_COLORS]

PACMAN_COLOR = formatColor(255.0 / 255.0, 255.0 / 255.0, 61.0 / 255)
PACMAN_SCALE = 0.5
# pacman_speed = 0.25

# Food
FOOD_COLOR = formatColor(1, 1, 1)
FOOD_SIZE = 0.1

# Laser
LASER_COLOR = formatColor(1, 0, 0)
LASER_SIZE = 0.02

# Capsule graphics
CAPSULE_COLOR = formatColor(1, 1, 1)
CAPSULE_SIZE = 0.25

# Drawing walls
WALL_RADIUS = 0.15


class InfoPane:
    def __init__(self, layout, gridSize, redTeam, blueTeam):
        self.gridSize = gridSize
        self.width = (layout.width) * gridSize
        self.base = (layout.height + 1) * gridSize
        self.height = INFO_PANE_HEIGHT
        self.fontSize = 24
        self.textColor = PACMAN_COLOR
        self.redTeam = redTeam
        self.blueTeam = blueTeam
        self.drawPane()

    def toScreen(self, pos, y=None):
        """
      Translates a point relative from the bottom left of the info pane.
    """
        if y == None:
            x, y = pos
        else:
            x = pos

        x = self.gridSize + x  # Margin
        y = self.base + y
        return x, y

    def drawPane(self):

        # Add the SCORE: xxx     TIME: xxx banner
        self.scoreText = create_text(self.toScreen(0, 0), self.textColor, self._scoreString(0), "Consolas",
                                     self.fontSize,
                                     "bold")

        self.timeText = create_text(self.toScreen(740, 0), self.textColor, self._timeString(1200), "Consolas",
                                    self.fontSize,
                                    "bold")

        # Add red team name on the left (besides SCORE:) with color TEAM_COLORS[0] (red)
        self.redText = create_text(self.toScreen(230, 0), TEAM_COLORS[0], self._redScoreString(), "Consolas",
                                   self.fontSize,
                                   "bold")

        # Add the "vs" word on the right of the red team name
        self.redText = create_text(self.toScreen(475, 0), self.textColor, "vs", "Consolas", self.fontSize, "bold")

        # Add red team name on the left (besides SCORE:) with color TEAM_COLORS[1] (blue)
        self.redText = create_text(self.toScreen(530, 0), TEAM_COLORS[1], self._blueScoreString(), "Consolas",
                                   self.fontSize,
                                   "bold")
        #
        # self.scoreText = text( self.toScreen(0, 0  ), self.textColor, self._infoString(0,1200), "Consolas", self.fontSize, "bold")
        # self.redText = text( self.toScreen(230, 0  ), TEAM_COLORS[0], self._redScoreString(), "Consolas", self.fontSize, "bold")
        # self.redText = text( self.toScreen(690, 0  ), TEAM_COLORS[1], self._blueScoreString(), "Consolas", self.fontSize, "bold")

    def _redScoreString(self):
        # return "RED: % 10s "%(self.redTeam[:12])
        return "%12s " % (self.redTeam[:12])

    def _blueScoreString(self):
        # return "BLUE: % 10s "%(self.blueTeam[:12])
        return "%-12s " % (self.blueTeam[:12])

    def updateRedText(self, score):
        change_text(self.redText, self._redScoreString())

    def updateBlueText(self, score):
        change_text(self.blueText, self._blueScoreString())

    def initializeghost_distances(self, distances):
        self.ghost_distance_text = []

        size = 20
        if self.width < 240:
            size = 12
        if self.width < 160:
            size = 10

        for i, d in enumerate(distances):
            t = create_text(self.toScreen(self.width / 2 + self.width / 8 * i, 0), GHOST_COLORS[i + 1], d, "Times",
                            size, "bold")
            self.ghost_distance_text.append(t)

    def _scoreString(self, score):
        return "SCORE: %2d" % (score)

    def _timeString(self, timeleft):
        return "TIME: %4d" % (timeleft)

    def updateScore(self, score, timeleft):
        change_text(self.scoreText, self._scoreString(score))
        change_text(self.timeText, self._timeString(timeleft))

    def setTeam(self, isBlue):
        text = "RED TEAM"
        if isBlue: text = "BLUE TEAM"
        self.teamText = text(self.toScreen(300, 0), self.textColor, text, "Times", self.fontSize, "bold")

    def updateghost_distances(self, distances):
        if len(distances) == 0: return
        if 'ghost_distance_text' not in dir(self):
            self.initializeghost_distances(distances)
        else:
            for i, d in enumerate(distances):
                change_text(self.ghost_distance_text[i], d)

    def drawGhost(self):
        pass

    def drawPacman(self):
        pass

    def drawWarning(self):
        pass

    def clearIcon(self):
        pass

    def updateMessage(self, message):
        pass

    def clearMessage(self):
        pass


class PacmanGraphics:
    def __init__(self, redTeam, redName, blueTeam, blueName, zoom=1.0, frameTime=0.0, capture=False):
        self.expanded_cells = []
        self.have_window = 0
        self.currentGhostImages = {}
        self.pacmanImage = None
        self.zoom = zoom
        self.gridSize = DEFAULT_GRID_SIZE * zoom
        self.capture = capture
        self.frameTime = frameTime
        self.redTeam = redTeam
        self.blueTeam = blueTeam

        self.redName = redTeam
        if redName:
            self.redName = redName
        self.blueName = blueTeam
        if blueName:
            self.blueName = blueName

        self.particleFilterImages = {0: [], 1: [], 2: [], 3: []}

    def initialize(self, state, isBlue=False):
        self.isBlue = isBlue
        self.startGraphics(state)

        # self.drawDistributions(state)
        self.distributionImages = None  # Initialized lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state

    def startGraphics(self, state):
        self.layout = state.layout
        layout = self.layout
        self.width = layout.width
        self.height = layout.height
        self.make_window(self.width, self.height)
        self.infoPane = InfoPane(layout, self.gridSize, self.redName, self.blueName)
        self.currentState = layout

    def drawDistributions(self, state):
        walls = state.layout.walls
        dist = []
        for x in range(walls.width):
            distx = []
            dist.append(distx)
            for y in range(walls.height):
                (screen_x, screen_y) = self.to_screen((x, y))
                block = square((screen_x, screen_y),
                               0.5 * self.gridSize,
                               color=BACKGROUND_COLOR,
                               filled=1, behind=2)
                distx.append(block)
        self.distributionImages = dist

    def drawStaticObjects(self, state):
        layout = self.layout
        self.drawWalls(layout.walls)
        self.food = self.drawFood(layout.food)
        self.capsules = self.drawCapsules(layout.capsules)
        refresh()

    def drawAgentObjects(self, state):
        self.agentImages = []  # (agentState, image)
        for index, agent in enumerate(state.agent_states):
            if agent.is_pacman:
                image = self.drawPacman(agent, index)
                self.agentImages.append((agent, image))
            else:
                image = self.drawGhost(agent, index)
                self.agentImages.append((agent, image))
        refresh()

    def swapImages(self, agentIndex, newState):
        """
      Changes an image from a ghost to a pacman or vis versa (for capture)
    """
        prevState, prevImage = self.agentImages[agentIndex]
        for item in prevImage: remove_from_screen(item)
        if newState.is_pacman:
            image = self.drawPacman(newState, agentIndex)
            self.agentImages[agentIndex] = (newState, image)
        else:
            image = self.drawGhost(newState, agentIndex)
            self.agentImages[agentIndex] = (newState, image)
        refresh()

    def update(self, newState):
        agentIndex = newState._agent_moved
        agentState = newState.agent_states[agentIndex]

        if self.agentImages[agentIndex][0].is_pacman != agentState.is_pacman: self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        if agentState.is_pacman:
            self.animatePacman(agentState, prevState, prevImage)
        else:
            self.moveGhost(agentState, agentIndex, prevState, prevImage)
        self.agentImages[agentIndex] = (agentState, prevImage)

        if newState._food_eaten != None:
            self.removeFood(newState._food_eaten, self.food)
        if newState._capsule_eaten != None:
            self.removeCapsule(newState._capsule_eaten, self.capsules)
        # dumping food
        if newState._food_added != None:
            for foodPos in newState._food_added:
                self.addFood(foodPos, self.food, newState.layout)

        self.infoPane.updateScore(newState.score, newState.timeleft)
        if 'ghost_distances' in dir(newState):
            self.infoPane.updateghost_distances(newState.ghost_distances)

        self.animateEnemyPositionParticleFilters()

    def animateEnemyPositionParticleFilters(self):
        DRAW_PARTICLES = False
        DRAW_CONDENSED_POSITION_DISTRIBUTION = False#not DRAW_PARTICLES

        for particle_filter_dict in [self.redEnemyPositionParticleFilters, self.blueEnemyPositionParticleFilters]:
            if particle_filter_dict is not None:
                for enemy, particle_filter in particle_filter_dict.items():
                    # delete old images
                    old_images, self.particleFilterImages[enemy] = self.particleFilterImages[enemy][:], []
                    for particle_image in old_images:
                        remove_from_screen(particle_image)

                    if DRAW_PARTICLES:
                        # draw new particles
                        self.drawParticleFilterParticles(particle_filter)
                    
                    if DRAW_CONDENSED_POSITION_DISTRIBUTION:
                        self.drawParticleFilterCondensedPositionDistribution(particle_filter)
        refresh()

    def drawParticleFilterParticles(self, particle_filter):
        # draw new particles
        particles = particle_filter.particles
        enemy = particle_filter.tracked_enemy_index
        particle_counter = Counter()
        for pos in particles:
            particle_counter[tuple(pos)] += 1
        
        for pos, n in particle_counter.items():
            screen_pos = self.to_screen(pos)
            MIN_PARTICLE_SIZE = 0.05
            MAX_PARTICLE_SIZE = 1
            particle_size = MIN_PARTICLE_SIZE + (MAX_PARTICLE_SIZE - MIN_PARTICLE_SIZE) * ((n/particle_filter.num_particles) ** 1/2)
            scaled_particle_size = self.gridSize * particle_size
            particle_image = circle(screen_pos,
                    scaled_particle_size,
                    outlineColor='#ffffff', fillColor=GHOST_COLORS[enemy],
                    width=0.6 * scaled_particle_size,style='chord')
            self.particleFilterImages[enemy].append(particle_image)
        current_estimate = particle_filter.estimate_position()
        current_estimate_screen_pos = self.to_screen(current_estimate)
        current_estimate_marker = create_text(current_estimate_screen_pos, GHOST_COLORS[enemy], 'X', size=24, anchor='center')
        self.particleFilterImages[enemy].append(current_estimate_marker)


    def drawParticleFilterCondensedPositionDistribution(self, particle_filter):
        # draw new particles
        position_distribution = particle_filter._EnemyPositionParticleFilter__get_condensed_position_distribution()
        enemy = particle_filter.tracked_enemy_index
        max_prob = position_distribution.max()
        MIN_SIZE = 0.05
        MAX_SIZE = 1
        nonzero_x, nonzero_y = position_distribution.nonzero()
        for x, y in zip(nonzero_x, nonzero_y):
                prob = position_distribution[x, y]
                
                screen_pos = self.to_screen((x, y))
                position_size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * ((prob/max_prob) ** 1/2)
                scaled_position_size = self.gridSize * position_size
                position_image = circle(screen_pos,
                        scaled_position_size,
                        outlineColor='#ffffff', fillColor=GHOST_COLORS[enemy],
                        width=0.6 * scaled_position_size,style='chord')
                self.particleFilterImages[enemy].append(position_image)
        current_estimate = particle_filter.estimate_position()
        current_estimate_screen_pos = self.to_screen(current_estimate)
        current_estimate_marker = create_text(current_estimate_screen_pos, GHOST_COLORS[enemy], 'X', size=24, anchor='center')
        self.particleFilterImages[enemy].append(current_estimate_marker)

    def make_window(self, width, height):
        grid_width = (width - 1) * self.gridSize
        grid_height = (height - 1) * self.gridSize
        screen_width = 2 * self.gridSize + grid_width
        screen_height = 2 * self.gridSize + grid_height + INFO_PANE_HEIGHT

        begin_graphics(screen_width,
                       screen_height,
                       BACKGROUND_COLOR,
                       "AI4EDUC Pacman (based on CS188 Pacman)")

    def drawPacman(self, pacman, index):
        position = self.getPosition(pacman)
        screen_point = self.to_screen(position)
        endpoints = self.getEndpoints(self.getDirection(pacman))

        width = PACMAN_OUTLINE_WIDTH
        outlineColor = PACMAN_COLOR
        fillColor = PACMAN_COLOR

        if self.capture:
            outlineColor = TEAM_COLORS[index % 2]
            fillColor = GHOST_COLORS[index]
            width = PACMAN_CAPTURE_OUTLINE_WIDTH

        return [circle(screen_point, PACMAN_SCALE * self.gridSize,
                       fillColor=fillColor, outlineColor=outlineColor,
                       endpoints=endpoints,
                       width=width)]

    def getEndpoints(self, direction, position=(0, 0)):
        x, y = position
        pos = x - int(x) + y - int(y)
        width = 30 + 80 * math.sin(math.pi * pos)

        delta = width / 2
        if (direction == 'West'):
            endpoints = (180 + delta, 180 - delta)
        elif (direction == 'North'):
            endpoints = (90 + delta, 90 - delta)
        elif (direction == 'South'):
            endpoints = (270 + delta, 270 - delta)
        else:
            endpoints = (0 + delta, 0 - delta)
        return endpoints

    def movePacman(self, position, direction, image):
        screenPosition = self.to_screen(position)
        endpoints = self.getEndpoints(direction, position)
        r = PACMAN_SCALE * self.gridSize
        moveCircle(image[0], screenPosition, r, endpoints)
        refresh()

    def animatePacman(self, pacman, prevPacman, image):
        if self.frameTime < 0:
            print('Press any key to step forward, "q" to play')
            keys = wait_for_keys()
            if 'q' in keys:
                self.frameTime = 0.1
        if self.frameTime > 0.01 or self.frameTime < 0:
            start = time.time()
            fx, fy = self.getPosition(prevPacman)
            px, py = self.getPosition(pacman)
            frames = 4.0
            for i in range(1, int(frames) + 1):
                pos = px * i / frames + fx * (frames - i) / frames, py * i / frames + fy * (frames - i) / frames
                self.movePacman(pos, self.getDirection(pacman), image)
                refresh()
                sleep(abs(self.frameTime) / frames)
        else:
            self.movePacman(self.getPosition(pacman), self.getDirection(pacman), image)
        refresh()

    def getGhostColor(self, ghost, ghostIndex):
        if ghost.scared_timer > 0:
            return SCARED_COLOR
        else:
            return GHOST_COLORS[ghostIndex]

    def drawGhost(self, ghost, agentIndex):
        pos = self.getPosition(ghost)
        dir = self.getDirection(ghost)
        (screen_x, screen_y) = (self.to_screen(pos))
        coords = []
        for (x, y) in GHOST_SHAPE:
            coords.append((x * self.gridSize * GHOST_SIZE + screen_x, y * self.gridSize * GHOST_SIZE + screen_y))

        colour = self.getGhostColor(ghost, agentIndex)
        body = polygon(coords, colour, filled=1)
        WHITE = formatColor(1.0, 1.0, 1.0)
        BLACK = formatColor(0.0, 0.0, 0.0)

        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        leftEye = circle((screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx / 1.5),
                          screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5)), self.gridSize * GHOST_SIZE * 0.2,
                         WHITE, WHITE)
        rightEye = circle((screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx / 1.5),
                           screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5)), self.gridSize * GHOST_SIZE * 0.2,
                          WHITE, WHITE)
        leftPupil = circle(
            (screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
            self.gridSize * GHOST_SIZE * 0.08, BLACK, BLACK)
        rightPupil = circle(
            (screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
            self.gridSize * GHOST_SIZE * 0.08, BLACK, BLACK)
        ghostImageParts = []
        ghostImageParts.append(body)
        ghostImageParts.append(leftEye)
        ghostImageParts.append(rightEye)
        ghostImageParts.append(leftPupil)
        ghostImageParts.append(rightPupil)

        return ghostImageParts

    def moveEyes(self, pos, dir, eyes):
        (screen_x, screen_y) = (self.to_screen(pos))
        dx = 0
        dy = 0
        if dir == 'North':
            dy = -0.2
        if dir == 'South':
            dy = 0.2
        if dir == 'East':
            dx = 0.2
        if dir == 'West':
            dx = -0.2
        moveCircle(eyes[0], (screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx / 1.5),
                             screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5)),
                   self.gridSize * GHOST_SIZE * 0.2)
        moveCircle(eyes[1], (screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx / 1.5),
                             screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy / 1.5)),
                   self.gridSize * GHOST_SIZE * 0.2)
        moveCircle(eyes[2], (
        screen_x + self.gridSize * GHOST_SIZE * (-0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
                   self.gridSize * GHOST_SIZE * 0.08)
        moveCircle(eyes[3], (
        screen_x + self.gridSize * GHOST_SIZE * (0.3 + dx), screen_y - self.gridSize * GHOST_SIZE * (0.3 - dy)),
                   self.gridSize * GHOST_SIZE * 0.08)

    def moveGhost(self, ghost, ghostIndex, prevGhost, ghostImageParts):
        old_x, old_y = self.to_screen(self.getPosition(prevGhost))
        new_x, new_y = self.to_screen(self.getPosition(ghost))
        delta = new_x - old_x, new_y - old_y

        for ghostImagePart in ghostImageParts:
            move_by(ghostImagePart, delta, lift=True)
        refresh()

        if ghost.scared_timer > 0:
            color = SCARED_COLOR
        else:
            color = GHOST_COLORS[ghostIndex]
        edit(ghostImageParts[0], ('fill', color), ('outline', color))
        self.moveEyes(self.getPosition(ghost), self.getDirection(ghost), ghostImageParts[-4:])
        refresh()

    def getPosition(self, agentState):
        if agentState.configuration == None: return (-1000, -1000)
        return agentState.get_position()

    def getDirection(self, agentState):
        if agentState.configuration == None: return Directions.STOP
        return agentState.configuration.get_direction()

    def finish(self):
        end_graphics()

    def to_screen(self, point):
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.gridSize
        y = (self.height - y) * self.gridSize
        return (x, y)

    # Fixes some TK issue with off-center circles
    def to_screen2(self, point):
        (x, y) = point
        # y = self.height - y
        x = (x + 1) * self.gridSize
        y = (self.height - y) * self.gridSize
        return (x, y)

    def drawWalls(self, wallMatrix):
        wallColor = WALL_COLOR
        for xNum, x in enumerate(wallMatrix):
            if self.capture and (xNum * 2) < wallMatrix.width: wallColor = TEAM_COLORS[0]
            if self.capture and (xNum * 2) >= wallMatrix.width: wallColor = TEAM_COLORS[1]

            for yNum, cell in enumerate(x):
                if cell:  # There's a wall here
                    pos = (xNum, yNum)
                    screen = self.to_screen(pos)
                    screen2 = self.to_screen2(pos)

                    # draw each quadrant of the square based on adjacent walls
                    wIsWall = self.isWall(xNum - 1, yNum, wallMatrix)
                    eIsWall = self.isWall(xNum + 1, yNum, wallMatrix)
                    nIsWall = self.isWall(xNum, yNum + 1, wallMatrix)
                    sIsWall = self.isWall(xNum, yNum - 1, wallMatrix)
                    nwIsWall = self.isWall(xNum - 1, yNum + 1, wallMatrix)
                    swIsWall = self.isWall(xNum - 1, yNum - 1, wallMatrix)
                    neIsWall = self.isWall(xNum + 1, yNum + 1, wallMatrix)
                    seIsWall = self.isWall(xNum + 1, yNum - 1, wallMatrix)

                    # NE quadrant
                    if (not nIsWall) and (not eIsWall):
                        # inner circle
                        circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (0, 91), 'arc')
                    if (nIsWall) and (not eIsWall):
                        # vertical line
                        line(add(screen, (self.gridSize * WALL_RADIUS, 0)),
                             add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5) - 1)), wallColor)
                    if (not nIsWall) and (eIsWall):
                        # horizontal line
                        line(add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (-1) * WALL_RADIUS)), wallColor)
                    if (nIsWall) and (eIsWall) and (not neIsWall):
                        # outer circle
                        circle(add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
                               WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (180, 271), 'arc')
                        line(add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (-1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (-1) * WALL_RADIUS)), wallColor)
                        line(add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                             add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (-0.5))), wallColor)

                    # NW quadrant
                    if (not nIsWall) and (not wIsWall):
                        # inner circle
                        circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (90, 181), 'arc')
                    if (nIsWall) and (not wIsWall):
                        # vertical line
                        line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)),
                             add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-0.5) - 1)), wallColor)
                    if (not nIsWall) and (wIsWall):
                        # horizontal line
                        line(add(screen, (0, self.gridSize * (-1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * (-0.5) - 1, self.gridSize * (-1) * WALL_RADIUS)), wallColor)
                    if (nIsWall) and (wIsWall) and (not nwIsWall):
                        # outer circle
                        circle(add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS)),
                               WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (270, 361), 'arc')
                        line(add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (-1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * (-0.5), self.gridSize * (-1) * WALL_RADIUS)), wallColor)
                        line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-2) * WALL_RADIUS + 1)),
                             add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (-0.5))), wallColor)

                    # SE quadrant
                    if (not sIsWall) and (not eIsWall):
                        # inner circle
                        circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (270, 361), 'arc')
                    if (sIsWall) and (not eIsWall):
                        # vertical line
                        line(add(screen, (self.gridSize * WALL_RADIUS, 0)),
                             add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5) + 1)), wallColor)
                    if (not sIsWall) and (eIsWall):
                        # horizontal line
                        line(add(screen, (0, self.gridSize * (1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * 0.5 + 1, self.gridSize * (1) * WALL_RADIUS)), wallColor)
                    if (sIsWall) and (eIsWall) and (not seIsWall):
                        # outer circle
                        circle(add(screen2, (self.gridSize * 2 * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                               WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (90, 181), 'arc')
                        line(add(screen, (self.gridSize * 2 * WALL_RADIUS - 1, self.gridSize * (1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * 0.5, self.gridSize * (1) * WALL_RADIUS)), wallColor)
                        line(add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                             add(screen, (self.gridSize * WALL_RADIUS, self.gridSize * (0.5))), wallColor)

                    # SW quadrant
                    if (not sIsWall) and (not wIsWall):
                        # inner circle
                        circle(screen2, WALL_RADIUS * self.gridSize, wallColor, wallColor, (180, 271), 'arc')
                    if (sIsWall) and (not wIsWall):
                        # vertical line
                        line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, 0)),
                             add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (0.5) + 1)), wallColor)
                    if (not sIsWall) and (wIsWall):
                        # horizontal line
                        line(add(screen, (0, self.gridSize * (1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * (-0.5) - 1, self.gridSize * (1) * WALL_RADIUS)), wallColor)
                    if (sIsWall) and (wIsWall) and (not swIsWall):
                        # outer circle
                        circle(add(screen2, (self.gridSize * (-2) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS)),
                               WALL_RADIUS * self.gridSize - 1, wallColor, wallColor, (0, 91), 'arc')
                        line(add(screen, (self.gridSize * (-2) * WALL_RADIUS + 1, self.gridSize * (1) * WALL_RADIUS)),
                             add(screen, (self.gridSize * (-0.5), self.gridSize * (1) * WALL_RADIUS)), wallColor)
                        line(add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (2) * WALL_RADIUS - 1)),
                             add(screen, (self.gridSize * (-1) * WALL_RADIUS, self.gridSize * (0.5))), wallColor)

    def isWall(self, x, y, walls):
        if x < 0 or y < 0:
            return False
        if x >= walls.width or y >= walls.height:
            return False
        return walls[x][y]

    def drawFood(self, foodMatrix):
        foodImages = []
        color = FOOD_COLOR
        for xNum, x in enumerate(foodMatrix):
            if self.capture and (xNum * 2) < foodMatrix.width: color = TEAM_COLORS[0]
            if self.capture and (xNum * 2) >= foodMatrix.width: color = TEAM_COLORS[1]
            imageRow = []
            foodImages.append(imageRow)
            for yNum, cell in enumerate(x):
                if cell:  # There's food here
                    screen = self.to_screen((xNum, yNum))
                    dot = circle(screen,
                                 FOOD_SIZE * self.gridSize,
                                 outlineColor=color, fillColor=color,
                                 width=1)
                    imageRow.append(dot)
                else:
                    imageRow.append(None)
        return foodImages

    def drawCapsules(self, capsules):
        capsuleImages = {}
        for capsule in capsules:
            (screen_x, screen_y) = self.to_screen(capsule)
            dot = circle((screen_x, screen_y),
                         CAPSULE_SIZE * self.gridSize,
                         outlineColor=CAPSULE_COLOR,
                         fillColor=CAPSULE_COLOR,
                         width=1)
            capsuleImages[capsule] = dot
        return capsuleImages

    def removeFood(self, cell, foodImages):
        x, y = cell
        remove_from_screen(foodImages[x][y])

    def addFood(self, pos, foodImages, layout):
        # only called with capture / contest mode, so
        # assume its red for now
        x, y = pos
        color = TEAM_COLORS[0]
        if (x * 2) >= layout.width:
            color = TEAM_COLORS[1]

        screen = self.to_screen(pos)
        dot = circle(screen,
                     FOOD_SIZE * self.gridSize,
                     outlineColor=color,
                     fillColor=color,
                     width=1)
        foodImages[x][y] = dot
        pass

    def removeCapsule(self, cell, capsuleImages):
        x, y = cell
        remove_from_screen(capsuleImages[(x, y)])

    def draw_expanded_cells(self, cells):
        """
    Draws an overlay of expanded grid positions for search agents
    """
        n = float(len(cells))
        baseColor = [1.0, 0.0, 0.0]
        self.clear_expanded_cells()
        self.expanded_cells = []
        for k, cell in enumerate(cells):
            screenPos = self.to_screen(cell)
            cellColor = formatColor(*[(n - k) * c * .5 / n + .25 for c in baseColor])
            block = square(screenPos,
                           0.5 * self.gridSize,
                           color=cellColor,
                           filled=1, behind=2)
            self.expanded_cells.append(block)
            if self.frameTime < 0:
                refresh()

    def clearDebug(self):
        if 'expanded_cells' in dir(self) and len(self.expanded_cells) > 0:
            for cell in self.expanded_cells:
                remove_from_screen(cell)

    def debugDraw(self, cells, color=[1.0, 0.0, 0.0], clear=False):
        n = float(len(cells))
        if clear:
            self.clearDebug()
            self.expanded_cells = []

        for k, cell in enumerate(cells):
            screenPos = self.to_screen(cell)
            cellColor = formatColor(*color)
            block = square(screenPos,
                           0.5 * self.gridSize,
                           color=cellColor,
                           filled=1, behind=2)
            self.expanded_cells.append(block)
            if self.frameTime < 0:
                refresh()

    def clear_expanded_cells(self):
        if 'expanded_cells' in dir(self) and len(self.expanded_cells) > 0:
            for cell in self.expanded_cells:
                remove_from_screen(cell)

    def update_distributions(self, distributions):
        "Draws an agent's belief distributions"
        if self.distributionImages == None:
            self.drawDistributions(self.previousState)
        for x in range(len(self.distributionImages)):
            for y in range(len(self.distributionImages[0])):
                image = self.distributionImages[x][y]
                weights = [dist[(x, y)] for dist in distributions]

                if sum(weights) != 0:
                    pass
                # Fog of war
                color = [0.0, 0.0, 0.0]
                colors = GHOST_VEC_COLORS[1:]  # With Pacman
                if self.capture: colors = GHOST_VEC_COLORS
                for weight, gcolor in zip(weights, colors):
                    color = [min(1.0, c + 0.95 * g * weight ** .3) for c, g in zip(color, gcolor)]
                changeColor(image, formatColor(*color))
        refresh()


class FirstPersonPacmanGraphics(PacmanGraphics):
    def __init__(self, zoom=1.0, showGhosts=True, capture=False, frameTime=0):
        PacmanGraphics.__init__(self, zoom, frameTime=frameTime)
        self.showGhosts = showGhosts
        self.capture = capture

    def initialize(self, state, isBlue=False):

        self.isBlue = isBlue
        PacmanGraphics.startGraphics(self, state)
        # Initialize distribution images
        walls = state.layout.walls
        dist = []
        self.layout = state.layout

        # Draw the rest
        self.distributionImages = None  # initialize lazily
        self.drawStaticObjects(state)
        self.drawAgentObjects(state)

        # Information
        self.previousState = state

    def lookAhead(self, config, state):
        if config.get_direction() == 'Stop':
            return
        else:
            pass
            # Draw relevant ghosts
            allGhosts = state.getGhostStates()
            visibleGhosts = state.getVisibleGhosts()
            for i, ghost in enumerate(allGhosts):
                if ghost in visibleGhosts:
                    self.drawGhost(ghost, i)
                else:
                    self.currentGhostImages[i] = None

    def getGhostColor(self, ghost, ghostIndex):
        return GHOST_COLORS[ghostIndex]

    def getPosition(self, ghostState):
        if not self.showGhosts and not ghostState.is_pacman and ghostState.get_position()[1] > 1:
            return (-1000, -1000)
        else:
            return PacmanGraphics.getPosition(self, ghostState)


def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


# Saving graphical output
# -----------------------
# Note: to make an animated gif from this postscript output, try the command:
# convert -delay 7 -loop 1 -compress lzw -layers optimize frame* out.gif
# convert is part of imagemagick (freeware)

SAVE_POSTSCRIPT = False
POSTSCRIPT_OUTPUT_DIR = 'frames'
FRAME_NUMBER = 0
import os


def saveFrame():
    "Saves the current graphical output as a postscript file"
    global SAVE_POSTSCRIPT, FRAME_NUMBER, POSTSCRIPT_OUTPUT_DIR
    if not SAVE_POSTSCRIPT: return
    if not os.path.exists(POSTSCRIPT_OUTPUT_DIR): os.mkdir(POSTSCRIPT_OUTPUT_DIR)
    name = os.path.join(POSTSCRIPT_OUTPUT_DIR, 'frame_%08d.ps' % FRAME_NUMBER)
    FRAME_NUMBER += 1
    writePostscript(name)  # writes the current canvas
