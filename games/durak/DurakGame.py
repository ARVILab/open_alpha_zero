from __future__ import print_function
import sys
import copy
sys.path.append('..')
sys.path.append('durak')

from games.durak.durak import WorldDurak
import numpy as np

from core.interfaces.Game import Game

class DurakGame(Game):
    def __init__(self):
        self.trueboard = WorldDurak()

    def reset(self):
        self.trueboard = WorldDurak()

    def get_cur_player(self):
        player = self.trueboard.get_player()
        print(player)
        return player

    def get_players_num(self):
        return 2

    def get_action_size(self):
        return 37

    def get_observation_size(self):
        return (333, 1)

    def make_move(self, action):
        valid_moves = self.get_valid_moves()
        if not valid_moves[action]:
            print(action, [x for x in range(37) if valid_moves[x]], valid_moves)
            action = list(valid_moves).index(1)
            print("Error: Invalid action was choosen. Reverting to a random valid action.")
        gameend, player = self.trueboard.life_iteration(action)
        if gameend:
            if self.trueboard.isWon(player):
                return (1.0, player)
            else:
                return (-1.0, player)
        else:
            return (0.0, player)

    def get_valid_moves(self, player=None):
        return self.trueboard.get_valid_moves()

    def is_ended(self):
        return self.trueboard.isGameEnd()

    def is_draw(self):
        is_ended = self.trueboard.isGameEnd()
        if is_ended:
            if self.trueboard.isWon(0):
                return False
            if self.trueboard.isWon(1):
                return False
            return True
        return False

    def get_score(self, player):
        if self.trueboard.isGameEnd():
            if self.trueboard.isWon(player):
                return 1.0
            else:
                return -1.0
        return 0.0

    def get_observation(self, player):
        return self.trueboard.observe(player)

    def get_observation_str(self, observation):
        if isinstance(observation, np.ndarray):
            return observation.tostring()
        else:
            return str(observation)

    def get_display_str(self):
        self.trueboard.observe(player=-1, verbose=True)
        return ""

    def clone(self):
        cloned_game = DurakGame()
        cloned_game.trueboard = self.trueboard.deepcopy()
        return cloned_game

    def reset_unknown_states(self, player):
        self.trueboard.reset_hidden_state(player)
