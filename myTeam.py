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

from agents.offensiveSwitchAStarAgent import OffensiveSwitchAStarAgent
from agents.defensiveAStarAgent import DefensiveAStarAgent
from enemy_localization.ownFoodSupervisor import OwnFoodSupervisor
from enemy_localization.enemySuicideDetector import EnemySuicideDetector

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
