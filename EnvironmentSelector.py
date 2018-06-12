from games.checkers.CheckersGame import CheckersGame

from games.checkers.nnet.CheckersResNNet import CheckersResNNet

from games.checkers.agent.AgentAlphaBeta import CheckersAgentAlphaBeta
from games.checkers.agent.CheckersHumanAgent import CheckersHumanAgent

from games.tictactoe.TicTacToeGame import TicTacToeGame

from games.tictactoe.nnet.TicTacToeNNet import TicTacToeNNet

from games.tictactoe.agent.TicTacToeHumanAgent import TicTacToeHumanAgent

from games.durak.DurakGame import DurakGame

from games.durak.nnet.DurakNNet import DurakNNet

from games.durak.agent.DurakHumanAgent import DurakHumanAgent

from core.nnet.NNet import NNet

from core.agents.AgentNNet import AgentNNet
from core.agents.AgentMCTS import AgentMCTS
from core.agents.AgentRandom import AgentRandom

import GPUtil


class EnvironmentSelector():
    GAME_CHECKERS_DEFAULT = "checkers_environment_default"

    GAME_TICTACTOE_DEFAULT = "tictactoe_environment_default"

    GAME_DURAK_DEFAULT = "durak_environment_default"

    class AgentProfile():
        def __init__(self, game, agent_profile):
            self.game = game
            self.agent_profile = agent_profile

    # agent profiles

    CHECKERS_AGENT_ALPHA_BETA = AgentProfile(GAME_CHECKERS_DEFAULT,
                                             "checkers_agent_alpha_beta")
    CHECKERS_AGENT_RANDOM = AgentProfile(GAME_CHECKERS_DEFAULT,
                                         "checkers_agent_random")
    CHECKERS_AGENT_HUMAN = AgentProfile(GAME_CHECKERS_DEFAULT,
                                        "checkers_agent_human")

    CHECKERS_AGENT_TRAIN_RCNN_DEFAULT = AgentProfile(GAME_CHECKERS_DEFAULT,
                                                     "checkers_agent_train_rcnn_default")
    CHECKERS_AGENT_TEST_AGENT_RCNN_DEFAULT = AgentProfile(GAME_CHECKERS_DEFAULT,
                                                          "checkers_agent_test_agent_rcnn_default")

    CHECKERS_AGENT_TRAIN_RCNN_TPU = AgentProfile(GAME_CHECKERS_DEFAULT,
                                                 "checkers_agent_train_rcnn_tpu")

    CHECKERS_AGENT_TRAIN_RCNN_DISTRIBUTED = AgentProfile(GAME_CHECKERS_DEFAULT,
                                                         "checkers_agent_train_rcnn_distributed")
    CHECKERS_AGENT_TEST_AGENT_RCNN_DISTRIBUTED = AgentProfile(GAME_CHECKERS_DEFAULT,
                                                              "checkers_agent_test_agent_rcnn_distributed")

    TICTACTOE_AGENT_TRAIN = AgentProfile(GAME_TICTACTOE_DEFAULT, "tictactoe_agent_train_default")
    TICTACTOE_AGENT_RANDOM = AgentProfile(GAME_TICTACTOE_DEFAULT, "tictactoe_agent_random")
    TICTACTOE_AGENT_HUMAN = AgentProfile(GAME_TICTACTOE_DEFAULT, "tictactoe_agent_human")

    DURAK_AGENT_TRAIN = AgentProfile(GAME_DURAK_DEFAULT, "durak_agent_train_default")
    DURAK_AGENT_RANDOM = AgentProfile(GAME_DURAK_DEFAULT, "durak_agent_random")
    DURAK_AGENT_HUMAN = AgentProfile(GAME_DURAK_DEFAULT, "durak_agent_human")

    def __init__(self):
        super().__init__()

        self.game_mapping = {
            EnvironmentSelector.GAME_CHECKERS_DEFAULT: CheckersGame(8, history_n=7),
            EnvironmentSelector.GAME_TICTACTOE_DEFAULT: TicTacToeGame(),
            EnvironmentSelector.GAME_DURAK_DEFAULT: DurakGame(),
        }

        self.agent_builder_mapping = {
            EnvironmentSelector.CHECKERS_AGENT_ALPHA_BETA: self.build_basic_checkers_agent,
            EnvironmentSelector.CHECKERS_AGENT_RANDOM: self.build_basic_checkers_agent,
            EnvironmentSelector.CHECKERS_AGENT_HUMAN: self.build_basic_checkers_agent,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DEFAULT: self.build_native_checkers_rcnn_agent,
            EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DEFAULT: self.build_native_checkers_rcnn_agent,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_TPU: self.build_tpu_checkers_agent,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DISTRIBUTED: self.build_horovod_checkers_agent,
            EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DISTRIBUTED: self.build_horovod_checkers_agent,

            EnvironmentSelector.TICTACTOE_AGENT_TRAIN: self.build_tictactoe_train_agent,
            EnvironmentSelector.TICTACTOE_AGENT_RANDOM: self.build_tictactoe_agent,
            EnvironmentSelector.TICTACTOE_AGENT_HUMAN: self.build_tictactoe_agent,

            EnvironmentSelector.DURAK_AGENT_TRAIN: self.build_durak_train_agent,
            EnvironmentSelector.DURAK_AGENT_RANDOM: self.build_durak_agent,
            EnvironmentSelector.DURAK_AGENT_HUMAN: self.build_durak_agent,
        }

        self.agent_profiles = [
            EnvironmentSelector.CHECKERS_AGENT_ALPHA_BETA,
            EnvironmentSelector.CHECKERS_AGENT_RANDOM,
            EnvironmentSelector.CHECKERS_AGENT_HUMAN,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DEFAULT,
            EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DEFAULT,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_TPU,

            EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DISTRIBUTED,
            EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DISTRIBUTED,

            EnvironmentSelector.TICTACTOE_AGENT_TRAIN,
            EnvironmentSelector.TICTACTOE_AGENT_RANDOM,
            EnvironmentSelector.TICTACTOE_AGENT_HUMAN,

            EnvironmentSelector.DURAK_AGENT_TRAIN,
            EnvironmentSelector.DURAK_AGENT_RANDOM,
            EnvironmentSelector.DURAK_AGENT_HUMAN,
        ]

    def get_profile(self, agent_profile_str):
        for profile in self.agent_profiles:
            if profile.agent_profile == agent_profile_str:
                return profile

        print("Error: could not find an agent profile by the key: ", agent_profile_str)

        return None

    def get_agent(self, agent_profile_str, native_multi_gpu_enabled=False):
        agent_profile = self.get_profile(agent_profile_str)
        if agent_profile in self.agent_builder_mapping:
            return self.agent_builder_mapping[agent_profile](agent_profile,
                                                             native_multi_gpu_enabled=native_multi_gpu_enabled)

        print("Error: could not find an agent by the key: ", agent_profile_str)

        return None

    def get_game(self, game_profile):
        if game_profile in self.game_mapping:
            return self.game_mapping[game_profile]

        print("Error: could not find a game with profile: ", game_profile)

        return None

    def build_basic_checkers_agent(self, agent_profile, native_multi_gpu_enabled=False):
        if agent_profile == EnvironmentSelector.CHECKERS_AGENT_ALPHA_BETA:
            return CheckersAgentAlphaBeta()
        elif agent_profile == EnvironmentSelector.CHECKERS_AGENT_RANDOM:
            return AgentRandom()
        elif agent_profile == EnvironmentSelector.CHECKERS_AGENT_HUMAN:
            return CheckersHumanAgent()
        return None

    def build_native_checkers_rcnn_agent(self, agent_profile, native_multi_gpu_enabled=False):
        game = self.game_mapping[agent_profile.game]

        if not native_multi_gpu_enabled:
            nnet = CheckersResNNet(game.get_observation_size()[0], game.get_observation_size()[1],
                                   game.get_observation_size()[2], game.get_action_size())
        else:
            nnet = CheckersResNNet(game.get_observation_size()[0], game.get_observation_size()[1],
                                   game.get_observation_size()[2], game.get_action_size(),
                                   multi_gpu=True, multi_gpu_n=len(GPUtil.getGPUs()))

        agent_nnet = AgentNNet(nnet)

        if agent_profile == EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DEFAULT:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.EXPLORATION_RATE_INIT, numMCTSSims=1500,
                             max_predict_time=3, num_threads=1)
        elif agent_profile == EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DEFAULT:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.NO_EXPLORATION, numMCTSSims=1500,
                             max_predict_time=10, num_threads=2, verbose=True)
        else:
            return None

    def build_tpu_checkers_agent(self, agent_profile, native_multi_gpu_enabled=False):
        from games.checkers.nnet.CheckersResNNetTPU import CheckersResNNetTPU

        assert not native_multi_gpu_enabled, "ERROR: TPU NNet does not support native multi-gpu mode!"

        game = self.game_mapping[agent_profile.game]

        nnet = CheckersResNNetTPU(game.get_observation_size()[0], game.get_observation_size()[1],
                                  game.get_observation_size()[2], game.get_action_size())

        agent_nnet = AgentNNet(nnet)

        if agent_profile == EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_TPU:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.EXPLORATION_RATE_INIT, numMCTSSims=200,
                             max_predict_time=None, verbose=False, num_threads=1)
        else:
            return None

    def build_horovod_checkers_agent(self, agent_profile, native_multi_gpu_enabled=False):
        from games.checkers.nnet.CheckersResNNetDistributed import CheckersResNNetDistributed

        assert not native_multi_gpu_enabled, "ERROR: Horovod NNet does not support native multi-gpu mode!"

        game = self.game_mapping[agent_profile.game]

        nnet = CheckersResNNetDistributed(game.get_observation_size()[0], game.get_observation_size()[1],
                                          game.get_observation_size()[2], game.get_action_size(),
                                          horovod_distributed=True)

        agent_nnet = AgentNNet(nnet)

        if agent_profile == EnvironmentSelector.CHECKERS_AGENT_TRAIN_RCNN_DISTRIBUTED:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.EXPLORATION_RATE_INIT, numMCTSSims=1500,
                             max_predict_time=5)
        elif agent_profile == EnvironmentSelector.CHECKERS_AGENT_TEST_AGENT_RCNN_DISTRIBUTED:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.NO_EXPLORATION, numMCTSSims=1500,
                             max_predict_time=10)
        else:
            return None

    def build_tictactoe_train_agent(self, agent_profile, native_multi_gpu_enabled=False):

        game = self.game_mapping[agent_profile.game]

        nnet = TicTacToeNNet(game.get_observation_size()[0], game.get_observation_size()[1],
                             game.get_observation_size()[2], game.get_action_size())

        agent_nnet = AgentNNet(nnet)

        if agent_profile == EnvironmentSelector.TICTACTOE_AGENT_TRAIN:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.EXPLORATION_RATE_MEDIUM, numMCTSSims=100,
                             max_predict_time=10)
        return None

    def build_tictactoe_agent(self, agent_profile, native_multi_gpu_enabled=False):

        game = self.game_mapping[agent_profile.game]

        if agent_profile == EnvironmentSelector.TICTACTOE_AGENT_RANDOM:
            return AgentRandom()
        elif agent_profile == EnvironmentSelector.TICTACTOE_AGENT_HUMAN:
            return TicTacToeHumanAgent(game=game)
        return None

    def build_durak_train_agent(self, agent_profile, native_multi_gpu_enabled=False):

        game = self.game_mapping[agent_profile.game]

        nnet = DurakNNet(*game.get_observation_size(), 1, game.get_action_size())

        agent_nnet = AgentNNet(nnet)

        if agent_profile == EnvironmentSelector.DURAK_AGENT_TRAIN:
            return AgentMCTS(agent_nnet, exp_rate=AgentMCTS.EXPLORATION_RATE_MEDIUM, numMCTSSims=100,
                             max_predict_time=10)
        return None

    def build_durak_agent(self, agent_profile, native_multi_gpu_enabled=False):

        game = self.game_mapping[agent_profile.game]

        if agent_profile == EnvironmentSelector.DURAK_AGENT_RANDOM:
            return AgentRandom()
        elif agent_profile == EnvironmentSelector.DURAK_AGENT_HUMAN:
            return DurakHumanAgent(game=game)
        return None
