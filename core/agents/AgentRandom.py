import sys

sys.path.append('..')

import numpy as np
import math
import numpy as np

from core.interfaces.Agent import Agent
from core.interfaces.Game import Game

import random


class AgentRandom(Agent):
    def __init__(self):
        super().__init__(name="Random Agent")

    def predict(self, game, game_player):
        moves = game.get_valid_moves(game_player)

        actions = [0.0] * len(moves)

        for idx, move_value in enumerate(moves):
            random_ = float(move_value) * float(random.random())
            actions[idx] = random_

        bestA = np.argmax(actions)
        actions = [0] * len(actions)
        actions[bestA] = 1

        return actions, -1
