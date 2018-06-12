import sys

sys.path.append('..')

import random
import numpy as np
import socket
import json

from time import sleep

from core.interfaces.Agent import Agent


class CheckersHumanAgent(Agent):
    def __init__(self, game):
        super().__init__(name="Human via Terminal")
        self.game = game

    def prepare_to_game(self):
        pass

    def predict(self, game, game_player):
        valid = game.get_valid_moves(game_player)
        all_moves = game.board_impl.get_all_moves()

        print("\n" + game.board_impl.get_true_state_str())

        print("Possible moves are: ")

        for idx in range(len(valid)):
            if valid[idx]:
                print("", idx, " => ", all_moves[idx])

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