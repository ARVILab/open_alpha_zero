from __future__ import print_function

import sys

sys.path.append('..')

from core.interfaces.Game import Game

import copy

import random

import numpy as np


class TicTacToeGame(Game):

	def __init__(self):
		self.board = np.array([
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0]])
		self.current_player = 1
		self.players = {1 : 0, -1 : 1}

	def reset(self):
		self.board = np.array([
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0]])
		self.current_player = 1
		self.players = {1 : 0, -1 : 1}

	def get_cur_player(self):
		return self.players[self.current_player]

	def get_players_num(self):
		return 2

	def get_action_size(self):
		return 9

	def get_observation_size(self):
		return self.board[np.newaxis,:,:].shape

	def make_move(self, action):

		valid_moves = self.get_valid_moves(0)

		self.board[action // 3, action % 3] = self.current_player

		score = self.get_score(self.current_player)
		self.current_player = self.current_player * -1
		player = self.players[self.current_player]
		return score, player

	def get_valid_moves(self, player):

		return (np.ravel(self.board) == 0) + 0

	def is_ended(self):

		players = [-1, 1]

		if self._check_player_win(players[0]):
			return True
		if self._check_player_win(players[1]):
			return True

		return self.is_draw()

	def is_draw(self):

		if sum(np.ravel(self.board) == 0) == 0:
			return True
		return False

	def get_score(self, player):

		player_curr = 1 if player == 0 else -1
		player_enemy = player_curr * -1

		if self._check_player_win(player_curr):
			return 1.0
		if self._check_player_win(player_enemy):
			return -1.0
		return 0.0

	def get_observation(self, player):

		player_in = 1 if player == 0 else -1

		observation = self.board * player_in
		return observation[np.newaxis,:,:]

	def get_observation_str(self, observation):

		return np.array2string(observation)

	def get_display_str(self):

		board_display = [['X' if x > 0 else 'O' if x < 0 else '.' for x in i] for i in self.board]
		board_display = ["\t" + "  ".join(i) for i in board_display]
		board_display.insert(0, "\n")

		return "\n".join(board_display)

	def clone(self):

		cloned_game = TicTacToeGame()
		cloned_game.board = np.array(self.board)
		cloned_game.current_player = self.current_player
		cloned_game.players = self.players
		return cloned_game

	def reset_unknown_states(self, player):
		pass


	def _check_player_win(self, player):

		for i in range(3):
			if (sum(self.board[:, i] == player) == 3)\
					or (sum(self.board[i, :] == player) == 3):
				return True

		if (sum(self.board.diagonal() == player) == 3)\
				or (sum(np.rot90(self.board).diagonal() == player) == 3):
			return True
		return False


	def get_board(self):
		return self.board
