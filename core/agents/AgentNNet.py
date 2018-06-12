import sys

sys.path.append('..')

import numpy as np

from core.interfaces.Agent import Agent

from multiprocessing import connection, Pipe
from threading import Thread


class AgentNNet(Agent):
    def __init__(self, nnet, name="Agent NNet"):
        super().__init__(name=name)
        self.nnet = nnet

    def prepare_to_game(self):
        pass

    def disable_training_capability(self, temp_dir=None, optimize=True):
        self.nnet.disable_training_capability(temp_dir=temp_dir, optimize=optimize)

    def enable_training_capability(self):
        self.nnet.enable_training_capability()

    def predict(self, game, game_player):
        observation = game.get_observation(game_player)
        observation = observation[np.newaxis, :, :]

        return self.nnet.predict(observation)

    def save(self, path_to_file):
        self.nnet.save(path_to_file)

    def load(self, path_to_file):
        self.nnet.load(path_to_file)

    def train(self, examples, batch_size=2048, epochs=10, verbose=1):
        input_boards, target_pis, target_vs = list(zip(*examples))

        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)

        self.nnet.train(input_boards, target_pis, target_vs, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def clone(self):
        return AgentNNet(self.nnet, name=self.name)
