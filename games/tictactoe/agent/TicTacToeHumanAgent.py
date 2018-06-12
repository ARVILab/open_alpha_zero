import sys
sys.path.append('../')

import random
import numpy as np
import json

from time import sleep

from core.interfaces.Agent import Agent

class TicTacToeHumanAgent(Agent):

    def __init__(self, game):
        super().__init__(name="Human via Terminal")
        self.game = game

    def prepare_to_game(self):
        pass

    def predict(self, game, game_player):
        valid = game.get_valid_moves(game_player)
        all_moves = np.ones((3, 3), dtype=np.int8) * -1

        for i in range(len(valid)):
            if valid[i]:
                all_moves[i // 3][i % 3] = i
            else:
                all_moves[i // 3][i % 3] = -1

        print("\nPositive numbers - possible moves:\n {0}".format(all_moves))

        print("\nPlease, input the index of the move.")
        while True:
            idx = int(input())

            if valid[idx]:
                print(idx)
                break
            else:
                print("Invalid move index!\nPlease, input the index of the move.")

        actions = [0] * len(valid)
        actions[idx] = 1

        return actions, -1
