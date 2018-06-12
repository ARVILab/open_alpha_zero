from __future__ import print_function
import sys

sys.path.insert(0, '../..')

try:
    from checkers import CheckerBoard
    from checkers import BLACK_PLAYER, WHITE_PLAYER
except ImportError:
    from games.checkers.checkers import CheckerBoard
    from games.checkers.checkers import BLACK_PLAYER, WHITE_PLAYER

from core.interfaces.Game import Game

import numpy as np

import copy

from collections import deque


class CheckersGame(Game):
    def __init__(self, n, history_n=7, cloned=False):
        self.n = n
        self.history_n = history_n

        self.player_mapping = {
            0: BLACK_PLAYER,
            1: WHITE_PLAYER
        }

        self.actions = {}
        self.states_history = None

        self.black_own_history_queue = None
        self.black_enemy_history_queue = None
        self.white_own_history_queue = None
        self.white_enemy_history_queue = None

        if not cloned:
            self.reset()

            for idx, move in enumerate(self.board_impl.get_all_moves()):
                self.actions[idx] = move

    def reset(self):
        self.board_impl = CheckerBoard(self.n)

        self.states_history = {}

        self.black_own_history_queue = deque([], maxlen=self.history_n)
        self.black_enemy_history_queue = deque([], maxlen=self.history_n)
        self.white_own_history_queue = deque([], maxlen=self.history_n)
        self.white_enemy_history_queue = deque([], maxlen=self.history_n)

        initial_state = np.array(self.board_impl.get_true_state())

        initial_state_black_own_history = self.board_impl.get_state_matrix_own(BLACK_PLAYER)
        initial_state_black_enemy_history = self.board_impl.get_state_matrix_enemy(BLACK_PLAYER)
        initial_state_white_own_history = self.board_impl.get_state_matrix_own(WHITE_PLAYER)
        initial_state_white_enemy_history = self.board_impl.get_state_matrix_enemy(WHITE_PLAYER)

        for idx in range(self.history_n):
            self.black_own_history_queue.append(initial_state_black_own_history)
            self.black_enemy_history_queue.append(initial_state_black_enemy_history)
            self.white_own_history_queue.append(initial_state_white_own_history)
            self.white_enemy_history_queue.append(initial_state_white_enemy_history)

        self.has_repeated_states = False

    def clone(self):
        obj = CheckersGame(self.n, history_n=self.history_n, cloned=True)

        obj.board_impl = self.board_impl.clone()
        obj.states_history = copy.copy(self.states_history)

        obj.black_own_history_queue = copy.copy(self.black_own_history_queue)
        obj.black_enemy_history_queue = copy.copy(self.black_enemy_history_queue)
        obj.white_own_history_queue = copy.copy(self.white_own_history_queue)
        obj.white_enemy_history_queue = copy.copy(self.white_enemy_history_queue)

        obj.has_repeated_states = self.has_repeated_states
        obj.actions = self.actions

        return obj

    def get_cur_player(self):
        cur_player = self.board_impl.get_current_player()
        if cur_player == self.board_impl.BLACK_PLAYER:
            return 0
        else:
            return 1

    def get_players_num(self):
        return 2

    def get_action_size(self):
        return len(self.actions)

    def get_observation_size(self):
        if self.history_n != 0:
            return (self.history_n * 2, self.n, self.n)
        else:
            return (self.n, self.n)

    def make_move(self, action_idx):

        player = self.get_cur_player()

        assert 0 <= action_idx < len(self.actions), "Invalid action index"

        action = self.actions[action_idx]

        is_capturing_move = self.board_impl.make_move(action)

        state = np.array(self.board_impl.get_true_state())
        state_hash = state.tostring()

        self.black_own_history_queue.append(self.board_impl.get_state_matrix_own(BLACK_PLAYER))
        self.black_enemy_history_queue.append(self.board_impl.get_state_matrix_enemy(BLACK_PLAYER))

        self.white_own_history_queue.append(self.board_impl.get_state_matrix_own(WHITE_PLAYER))
        self.white_enemy_history_queue.append(self.board_impl.get_state_matrix_enemy(WHITE_PLAYER))

        if is_capturing_move:
            # clear states history for repeated states
            # since we don't need to check for the states
            # which cannot be repeated due to changed
            # num of pieces on the board
            self.states_history = {}
            self.has_repeated_states = False

        if state_hash in self.states_history:
            repeated_states = self.states_history[state_hash]
            self.states_history[state_hash] = repeated_states + 1
            self.has_repeated_states = True
        else:
            self.states_history[state_hash] = 1

        return self.get_score(player), self.get_cur_player()

    def get_valid_moves(self, player):
        possible_idx_actions = [0] * self.get_action_size()

        inner_player = self.player_mapping[player]

        possible_moves = self.board_impl.get_legal_moves(player=inner_player)

        # forbid repeated states
        for idx, action in self.actions.items():
            if action in possible_moves:
                possible_idx_actions[idx] = 1

                if self.has_repeated_states:
                    # simulate move
                    board_clone = self.board_impl.clone()
                    board_clone.set_current_player(inner_player)
                    board_clone.make_move(action)

                    state = np.array(board_clone.get_true_state())
                    state_hash = state.tostring()

                    if state_hash in self.states_history:
                        repeated_states = self.states_history[state_hash]
                        if repeated_states >= 2:
                            # Action forbidden due to the potential draw situation
                            possible_idx_actions[idx] = 0

        return np.array(possible_idx_actions)

    def is_ended(self):
        return self.is_draw() or np.sum(self.get_valid_moves(0)) == 0 or np.sum(self.get_valid_moves(1)) == 0

    def is_draw(self):
        return self.board_impl.is_draw()

    def get_score(self, player):
        if self.is_ended():
            if self.is_draw():
                return -1

            if np.sum(self.get_valid_moves(player)) == 0:
                return -1
            else:
                return 1

        return 0

    def get_observation(self, player):
        inner_player = self.player_mapping[player]

        if self.history_n == 0:
            observation = np.array(self.board_impl.get_observation(inner_player))
        else:
            if inner_player == BLACK_PLAYER:
                own_history = list(reversed(self.black_own_history_queue))
                enemy_history = list(reversed(self.black_enemy_history_queue))
            else:
                own_history = list(reversed(self.white_own_history_queue))
                enemy_history = list(reversed(self.white_enemy_history_queue))

            observation = []
            observation.extend(own_history)
            observation.extend(enemy_history)

            observation = np.array(observation)

        return observation

    def get_observation_str(self, observation):
        return observation.tostring()

    def get_display_str(self):
        # return self.board_impl.get_state_str()
        return self.board_impl.get_true_state_str()

    def reset_unknown_states(self, player):
        pass

    def _get_state(self):
        return np.array(self.board_impl.get_true_state())

    def get_custom_score(self, player):
        own_pieces, own_kings, enemy_pieces, enemy_kings = self.get_pieces(player)

        return own_pieces + 2 * own_kings - (enemy_pieces + 2 * enemy_kings)

    def get_pieces(self, player):
        inner_player = self.player_mapping[player]

        return self.board_impl.get_pieces(inner_player)
