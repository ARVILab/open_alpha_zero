from termcolor import colored
import random
import numpy as np
from math import copysign

import collections
import copy

DEFAULT_BOARD_SIZE = 8


def is_even(num):
    return num % 2 == 0


def is_game_piece(x, y):
    return (is_even(y) and (not is_even(x))) or ((not is_even(y)) and is_even(x))


BLACK_PLAYER, WHITE_PLAYER = 1, -1


class CheckerBoard:
    def __init__(self, n=None, cloned=False):
        if n is not None:
            # TODO implement STARTING WHITES and STARTING BLACKS
            self.board_n = n
        else:
            self.board_n = DEFAULT_BOARD_SIZE

        self.BLACK_PLAYER, self.WHITE_PLAYER = 1, -1
        self.EMPTY, self.BLACK, self.BLACK_KING, self.WHITE, self.WHITE_KING = 0, 1, 3, -1, -3

        INITIAL_BOARD = [
            [(1, 0), self.WHITE], [(3, 0), self.WHITE], [(5, 0), self.WHITE], [(7, 0), self.WHITE],
            [(0, 1), self.WHITE], [(2, 1), self.WHITE], [(4, 1), self.WHITE], [(6, 1), self.WHITE],
            [(1, 2), self.WHITE], [(3, 2), self.WHITE], [(5, 2), self.WHITE], [(7, 2), self.WHITE],
            [(0, 5), self.BLACK], [(2, 5), self.BLACK], [(4, 5), self.BLACK], [(6, 5), self.BLACK],
            [(1, 6), self.BLACK], [(3, 6), self.BLACK], [(5, 6), self.BLACK], [(7, 6), self.BLACK],
            [(0, 7), self.BLACK], [(2, 7), self.BLACK], [(4, 7), self.BLACK], [(6, 7), self.BLACK]
        ]

        # TODO debug
        # for debugging purposes
        # INITIAL_BOARD = [
        #     [(1, 0), self.EMPTY], [(3, 0), self.EMPTY], [(5, 0), self.EMPTY], [(7, 0), self.EMPTY],
        #     [(0, 1), self.EMPTY], [(2, 1), self.EMPTY], [(4, 1), self.EMPTY], [(6, 1), self.EMPTY],
        #     [(1, 2), self.EMPTY], [(3, 2), self.EMPTY], [(5, 2), self.EMPTY], [(7, 2), self.EMPTY],
        #     [(0, 3), self.EMPTY], [(2, 3), self.EMPTY], [(4, 3), self.EMPTY], [(6, 3), self.EMPTY],
        #     [(1, 4), self.EMPTY], [(3, 4), self.EMPTY], [(5, 4), self.EMPTY], [(7, 4), self.EMPTY],
        #     [(0, 5), self.EMPTY], [(2, 5), self.EMPTY], [(4, 5), self.EMPTY], [(6, 5), self.EMPTY],
        #     [(1, 6), self.EMPTY], [(3, 6), self.EMPTY], [(5, 6), self.EMPTY], [(7, 6), self.EMPTY],
        #     [(0, 7), self.EMPTY], [(2, 7), self.EMPTY], [(4, 7), self.EMPTY], [(6, 7), self.EMPTY]
        # ]

        self.bw_state = None
        self.wb_state = None

        self.current_player = self.BLACK_PLAYER

        self.position_to_continue_jumps = None
        self.player_jumps = None
        self.all_moves = None

        if not cloned:
            self.bw_state = [x[:] for x in [[self.EMPTY] * self.board_n] * self.board_n]
            self.wb_state = [x[:] for x in [[self.EMPTY] * self.board_n] * self.board_n]

            for piece_obj in INITIAL_BOARD:
                position = piece_obj[0]
                piece = piece_obj[1]
                self.bw_state[position[1]][position[0]] = piece
                self.wb_state[7 - position[1]][7 - position[0]] = piece * (-1)

            self.get_all_moves()

    def get_all_moves(self):
        if self.all_moves is None:
            moves = []

            for y, row in enumerate(self.bw_state):
                for x, piece in enumerate(row):
                    if is_game_piece(x, y):
                        position = (x, y)
                        moves.extend(self._get_all_moves_from(position))

            self.all_moves = moves

        return self.all_moves

    def clone(self):
        obj = CheckerBoard(self.board_n, cloned=True)

        obj.bw_state = copy.deepcopy(self.bw_state)
        obj.wb_state = copy.deepcopy(self.wb_state)

        obj.current_player = self.current_player
        obj.player_jumps = self.player_jumps
        obj.position_to_continue_jumps = self.position_to_continue_jumps
        obj.all_moves = self.all_moves

        return obj

    def set_state(self, state, player_perspective):
        if player_perspective == self.BLACK_PLAYER:
            self.bw_state = state
        else:
            self.wb_state = state

        self._update_state(player_perspective)

    def get_true_state(self):
        return self.get_observation(self.BLACK_PLAYER)

    def get_observation(self, player):
        if player == self.BLACK_PLAYER:
            return self.bw_state
        else:
            return self.wb_state

    def get_state_matrix_own(self, player):
        if player == self.BLACK_PLAYER:
            state = copy.deepcopy(self.bw_state)
        else:
            state = copy.deepcopy(self.wb_state)

        for y, row in enumerate(state):
            for x, piece in enumerate(row):
                if self.is_enemy_piece(piece):
                    state[y][x] = self.EMPTY

        return state

    def get_state_matrix_enemy(self, player):
        if player == self.BLACK_PLAYER:
            state = copy.deepcopy(self.bw_state)
        else:
            state = copy.deepcopy(self.wb_state)

        for y, row in enumerate(state):
            for x, piece in enumerate(row):
                if self.is_own_piece(piece):
                    state[y][x] = self.EMPTY

        return state

    def is_own_piece(self, piece):
        return piece == self.BLACK or piece == self.BLACK_KING

    def is_enemy_piece(self, piece):
        return piece == self.WHITE or piece == self.WHITE_KING

    def get_legal_moves(self, player, debug=False):
        observation = self.get_observation(player)

        if debug:
            print("observation for player %d \n" % player, observation)

        if self.position_to_continue_jumps is not None \
                and self.player_jumps is not None \
                and self.player_jumps == player:
            if debug:
                print("continue jumps")
            return self.get_capturing_moves_from(self.position_to_continue_jumps, observation)

        capturing_moves = []

        for y, row in enumerate(observation):
            for x, piece in enumerate(row):
                position = (x, y)
                if self.is_own_piece(piece):
                    if debug:
                        print("own piece ", position)
                    capturing_moves_from = self.get_capturing_moves_from(position, observation)
                    if len(capturing_moves_from) > 0:
                        capturing_moves.extend(capturing_moves_from)

        if len(capturing_moves) > 0:
            return capturing_moves

        regular_moves = []

        for y, row in enumerate(observation):
            for x, piece in enumerate(row):
                position = (x, y)
                if self.is_own_piece(piece):
                    regular_moves_from = self.get_regular_moves_from(position, observation)
                    if len(regular_moves_from) > 0:
                        regular_moves.extend(regular_moves_from)

        return regular_moves

    def get_current_player(self):
        return self.current_player

    def set_current_player(self, player):
        self.current_player = player

    def make_move(self, move):
        assert len(move) == 2 or len(move[0]) == 2 or len(move[1]) == 2, "Invalid move format"

        cur_player = self.get_current_player()
        observation = self.get_observation(cur_player)

        start_position = move[0]
        end_position = move[1]

        end_x = end_position[0]
        end_y = end_position[1]

        start_x = start_position[0]
        start_y = start_position[1]

        move_x = end_x - start_x
        move_y = end_y - start_y

        capturing_move = abs(move_x) > 1 and abs(move_y) > 1

        start_piece = observation[start_y][start_x]
        end_piece = observation[end_y][end_x]

        if capturing_move:
            # capturing move detected
            captured_piece_x = int(start_x + copysign(1, move_x))
            captured_piece_y = int(start_y + copysign(1, move_y))

            captured_piece = observation[captured_piece_y][captured_piece_x]

            assert not (self.is_own_piece(captured_piece) or captured_piece == self.EMPTY), "Invalid capturing move"

            observation[captured_piece_y][captured_piece_x] = self.EMPTY

        if end_piece != self.EMPTY:
            print("observation: \n", observation)

        assert end_piece == self.EMPTY, "Invalid move: end position is not empty"

        observation[start_y][start_x] = self.EMPTY

        if end_y == 0:
            observation[end_y][end_x] = self.BLACK_KING
        else:
            observation[end_y][end_x] = start_piece

        self._update_state(cur_player)

        if capturing_move and len(self.get_capturing_moves_from(end_position, observation)) > 0:
            self.position_to_continue_jumps = end_position
            self.player_jumps = cur_player
        else:
            self.position_to_continue_jumps = None
            self.player_jumps = None
            self.switch_player()

        return capturing_move

    def switch_player(self):
        if self.current_player == self.BLACK_PLAYER:
            self.current_player = self.WHITE_PLAYER
        else:
            self.current_player = self.BLACK_PLAYER

    def __str__(self):
        return self.get_true_state_str()

    def get_state_str(self):
        str = ""
        if self.get_current_player() == BLACK_PLAYER:
            state = self.bw_state
        else:
            state = self.wb_state

        for y, row in enumerate(state):
            for x, piece in enumerate(row):
                id = " (%d,%d)E " % (x, y)
                man = " (%d,%d)M " % (x, y)
                king = " (%d,%d)K " % (x, y)

                if piece == self.BLACK:
                    id = colored(man, 'red')
                elif piece == self.BLACK_KING:
                    id = colored(king, 'red')
                elif piece == self.WHITE:
                    id = colored(man, 'green')
                elif piece == self.WHITE_KING:
                    id = colored(king, 'green')
                str = str + id
            str = str + "\n"
        return str

    def get_true_state_str(self):
        str = ""
        for y, row in enumerate(self.bw_state):
            for x, piece in enumerate(row):
                id = " (%d,%d)E " % (x, y)
                man = " (%d,%d)M " % (x, y)
                king = " (%d,%d)K " % (x, y)

                if piece == self.BLACK:
                    id = colored(man, 'red')
                elif piece == self.BLACK_KING:
                    id = colored(king, 'red')
                elif piece == self.WHITE:
                    id = colored(man, 'green')
                elif piece == self.WHITE_KING:
                    id = colored(king, 'green')
                str = str + id
            str = str + "\n"
        return str

    # private

    def _update_state(self, updated_player):
        if updated_player == self.BLACK_PLAYER:
            for y, row in enumerate(self.bw_state):
                for x, piece in enumerate(row):
                    self.wb_state[7 - y][7 - x] = piece * -1
        else:
            for y, row in enumerate(self.wb_state):
                for x, piece in enumerate(row):
                    self.bw_state[7 - y][7 - x] = piece * -1

    def _get_all_moves_from(self, position):
        end_positions = [
            (position[0] - 1, position[1] - 1),
            (position[0] - 1, position[1] + 1),
            (position[0] + 1, position[1] - 1),
            (position[0] + 1, position[1] + 1),
            (position[0] - 2, position[1] - 2),
            (position[0] - 2, position[1] + 2),
            (position[0] + 2, position[1] - 2),
            (position[0] + 2, position[1] + 2),
        ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < self.board_n and 0 <= end_position[1] < self.board_n:
                moves.append([position, end_position])

        return moves

    def get_regular_moves_from(self, position, observation):
        piece = observation[position[1]][position[0]]

        end_positions = []
        if piece == self.BLACK:
            end_positions = [
                (position[0] - 1, position[1] - 1),
                (position[0] + 1, position[1] - 1),
            ]
        elif piece == self.WHITE:
            end_positions = [
                (position[0] - 1, position[1] + 1),
                (position[0] + 1, position[1] + 1),
            ]
        else:
            end_positions = [
                (position[0] - 1, position[1] - 1),
                (position[0] - 1, position[1] + 1),
                (position[0] + 1, position[1] - 1),
                (position[0] + 1, position[1] + 1),
            ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < self.board_n and 0 <= end_position[1] < self.board_n:
                piece = observation[end_position[1]][end_position[0]]
                if piece == self.EMPTY:
                    moves.append([position, end_position])

        return moves

    def get_capturing_moves_from(self, position, observation):

        piece = observation[position[1]][position[0]]

        end_positions = []
        if piece == self.BLACK:
            end_positions = [
                (position[0] - 2, position[1] - 2),
                (position[0] + 2, position[1] - 2),
            ]
        elif piece == self.WHITE:
            end_positions = [
                (position[0] - 2, position[1] + 2),
                (position[0] + 2, position[1] + 2),
            ]
        else:
            end_positions = [
                (position[0] - 2, position[1] - 2),
                (position[0] - 2, position[1] + 2),
                (position[0] + 2, position[1] - 2),
                (position[0] + 2, position[1] + 2),
            ]

        moves = []

        for end_position in end_positions:
            if 0 <= end_position[0] < self.board_n and 0 <= end_position[1] < self.board_n:

                piece = observation[end_position[1]][end_position[0]]

                if piece == self.EMPTY:
                    move_x = end_position[0] - position[0]
                    move_y = end_position[1] - position[1]

                    captured_piece_x = int(position[0] + copysign(1, move_x))
                    captured_piece_y = int(position[1] + copysign(1, move_y))

                    captured_piece = observation[captured_piece_y][captured_piece_x]

                    if self.is_enemy_piece(captured_piece):
                        moves.append([position, end_position])

        return moves

    def is_draw(self):
        own_pieces, own_kings, enemy_pieces, enemy_kings = self.get_pieces(self.BLACK_PLAYER)

        return own_kings == 1 and enemy_kings == 1 and own_pieces == 0 and enemy_pieces == 0

    def get_pieces(self, player):

        observation = self.get_observation(player)

        own_pieces = 0
        own_kings = 0
        enemy_pieces = 0
        enemy_kings = 0

        for y, row in enumerate(observation):
            for x, piece in enumerate(row):
                if piece == self.BLACK:
                    own_pieces += 1
                elif piece == self.BLACK_KING:
                    own_kings += 1
                elif piece == self.WHITE:
                    enemy_pieces += 1
                elif piece == self.WHITE_KING:
                    enemy_kings += 1

        return own_pieces, own_kings, enemy_pieces, enemy_kings
