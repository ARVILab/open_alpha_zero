import sys

sys.path.append('..')

import numpy as np
import math
import numpy as np

from core.interfaces.Agent import Agent
from core.interfaces.Game import Game

import time


class CheckersAgentAlphaBeta(Agent):
    def __init__(self, depth=6, name="Agent AlphaBeta"):
        super().__init__(name=name)
        self.depth = depth

    def predict(self, game, game_player):

        self.ab_num = 0

        assert (game.get_cur_player() == game_player, "Cur player must be game player!")

        start_predict = time.time()

        value, best_action = self.alpha_beta(game, self.depth, float('-inf'), float('inf'), True)

        assert (best_action is not None, "Best action cannot be None!")

        probs = [0] * game.get_action_size()
        probs[best_action] = 1

        end_predict = time.time()
        print("predict Alpha Beta", end_predict - start_predict)

        print("ab num: ", self.ab_num)

        return probs, value

    def clone(self):
        return CheckersAgentAlphaBeta()

    def alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        """
        A method implementing alpha-beta pruning to decide what move to make given 
        the current board configuration. 
        """
        self.ab_num += 1

        cur_player = game.get_cur_player()

        if game.is_ended():
            if game.is_draw():
                # terminal node
                return 0, None

            if game.get_score(cur_player) < 0:
                if maximizing_player:
                    # Using integers instead of float("inf") so it's less than float("inf") not equal to
                    return -10000000, None
                else:
                    return 10000000, None
            else:
                if maximizing_player:
                    return 1000000, None
                else:
                    return -1000000, None

        if depth == 0:
            if maximizing_player:
                return game.get_custom_score(cur_player), None
            else:
                return -game.get_custom_score(cur_player), None

        desired_move_index = None
        valid_actions = game.get_valid_moves(cur_player)

        if maximizing_player:
            v = float('-inf')
            for action in range(game.get_action_size()):
                if valid_actions[action]:
                    game_cloned = game.clone()
                    _, next_player = game_cloned.make_move(action)

                    alpha_beta_results = self.alpha_beta(game_cloned, depth - 1, alpha, beta, cur_player == next_player)
                    if v < alpha_beta_results[0]:
                        v = alpha_beta_results[0]
                        alpha = max(alpha, v)
                        desired_move_index = action
                    if beta <= alpha:
                        break
            if desired_move_index is None:
                return v, None
            return v, desired_move_index
        else:
            v = float('inf')
            for action in range(game.get_action_size()):
                if valid_actions[action]:
                    game_cloned = game.clone()
                    _, next_player = game_cloned.make_move(action)
                    alpha_beta_results = self.alpha_beta(game_cloned, depth - 1, alpha, beta, cur_player != next_player)

                    if v > alpha_beta_results[0]:
                        v = alpha_beta_results[0]
                        desired_move_index = action
                        beta = min(beta, v)
                    if beta <= alpha:
                        break

            if desired_move_index is None:
                return v, None
            return v, desired_move_index
