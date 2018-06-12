class Agent():

    def __init__(self, name="Agent"):
        self.name = name

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def prepare_to_game(self):
        pass

    def predict(self, game, game_player):
        pass

    def on_turn_finished(self, game):
        pass

    def save(self, path_to_file):
        pass

    def load(self, path_to_file):
        pass

    def train(self, train_examples, batch_size = 2048, epochs = 10, verbose = 1):
        pass

    def clone(self):
        pass

    def set_exploration_enabled(self, enabled):
        pass