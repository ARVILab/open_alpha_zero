import sys
sys.path.append('..')

import random
import numpy as np

from core.interfaces.Agent import Agent
from core.interfaces.Game import Game

class DurakHumanAgent(Agent):
    def __init__(self, game):
        super().__init__(name= "HUMAN DURAK PLAYER")
        self.game = game

    def predict(self, game: Game, game_player):

        valid = self.game.get_valid_moves(game_player)
        all_moves = self.game.get_valid_moves(game_player)

        observation = self.game.trueboard.observe(-1, verbose=True)

        print("Possible moves are: {0}".format([x for x in range(37) if all_moves[x]]))

        print("Please, input the index of the move.")
        while True:
            idx = int(input())

            if valid[idx]:
                break
            else:
                print('Invalid move index.')

        actions = [0] * len(valid)
        actions[idx] = 1

        return actions, -1
