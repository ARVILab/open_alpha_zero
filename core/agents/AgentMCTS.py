import sys

sys.path.append('..')

import numpy as np
import math
import numpy as np

from core.interfaces.Agent import Agent
from core.interfaces.Game import Game

EPS = 1e-8

import time

from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock
from collections import defaultdict


class AgentMCTS(Agent):
    NO_EXPLORATION = 0
    EXPLORATION_RATE_INIT = 1
    EXPLORATION_RATE_LOW = 1.5
    EXPLORATION_RATE_MEDIUM = 2
    EXPLORATION_RATE_HIGH = 2.5

    def __init__(self, agent,
                 name="Agent MCTS", exp_rate=EXPLORATION_RATE_INIT,
                 cpuct=1, numMCTSSims=10,
                 max_predict_time=None,
                 num_threads=1,
                 verbose=False):
        super().__init__(name=name)

        self.agent = agent

        self.prev_exp_rate = exp_rate
        self.exp_rate = exp_rate

        self.cpuct = cpuct
        self.numMCTSSims = numMCTSSims
        self.max_predict_time = max_predict_time
        self.num_threads = num_threads

        self.verbose = verbose

        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.node_lock = defaultdict(Lock)
        self.predict_executor = ThreadPoolExecutor(max_workers=num_threads)

    def set_exploration_rate(self, exp_rate):
        self.exp_rate = exp_rate

    def disable_training_capability(self, temp_dir=None, optimize=True):
        self.agent.disable_training_capability(temp_dir=temp_dir, optimize=optimize)

    def enable_training_capability(self):
        self.agent.enable_training_capability()

    def predict(self, game, game_player):
        self.simulations_num = 0
        self.nnet_spent = 0
        start_predict = time.time()

        canonical_state = game.get_observation(game_player)
        canonical_state_str = game.get_observation_str(canonical_state)

        if self.num_threads == 1:
            value = self.simulate_sync(game, game_player, canonical_state_str)
        else:
            value = self.simulate_async(game, game_player, canonical_state_str)

        counts = [self.Nsa[(canonical_state_str, a)] if (canonical_state_str, a) in self.Nsa else 0 for a in
                  range(game.get_action_size())]

        if self.exp_rate == AgentMCTS.NO_EXPLORATION:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
        else:
            counts = [x ** (1. / self.exp_rate) for x in counts]
            probs = [x / float(sum(counts)) for x in counts]

        if sum(probs) == 0:
            print("Warning: probabilities sum up to 0!")

        if self.verbose:
            print("MCTS simulated %d new states in %f seconds." % (self.simulations_num, (time.time() - start_predict)))
            print("NNet spent time: %f. Average: %f" % (self.nnet_spent, (self.nnet_spent / self.simulations_num)))

        return probs, value

    def save(self, path_to_file):
        self.agent.save(path_to_file)

    def load(self, path_to_file):
        self.agent.load(path_to_file)

    def save_model(self, path_to_file):
        self.agent.save_model(path_to_file)

    def load_model(self, path_to_file):
        self.agent.load_model(path_to_file)

    def train(self, train_examples, batch_size=2048, epochs=10, verbose=1):
        self.agent.train(train_examples, batch_size=batch_size, epochs=epochs, verbose=verbose)

    def clone(self):
        return AgentMCTS(self.agent,
                         name=self.name,
                         exp_rate=self.prev_exp_rate,
                         cpuct=self.cpuct,
                         numMCTSSims=self.numMCTSSims,
                         max_predict_time=self.max_predict_time,
                         num_threads=self.num_threads,
                         verbose=self.verbose)

    def prepare_to_game(self):
        self.reset_search_tree()

    def reset_search_tree(self):
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}

    def set_exploration_enabled(self, enabled):
        if enabled:
            self.exp_rate = self.prev_exp_rate
        else:
            self.exp_rate = AgentMCTS.NO_EXPLORATION

    def simulate_async(self, game, game_player, canonical_state_str):
        start_predict = time.time()

        futures = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for idx in range(self.numMCTSSims):
                game_clone = game.clone()
                game_clone.reset_unknown_states(game_player)

                if idx == 0:
                    # sanity check
                    clone_canonical_state = game_clone.get_observation(game_player)
                    clone_canonical_state_str = game_clone.get_observation_str(clone_canonical_state)

                    if canonical_state_str != clone_canonical_state_str:
                        print("Warning: states are not equal after reset_unknown_states()")

                futures.append(executor.submit(self.search_async, game_clone, game_player))

            cancelled = False

            for idx, future in enumerate(futures):
                if cancelled:
                    future.cancel()
                    continue

                start_sim_time = time.time()

                _, value = future.result()

                end_sim_time = time.time()

                if self.max_predict_time and idx != 0:
                    next_sim_end_estimation = end_sim_time + (end_sim_time - start_sim_time)
                    if (next_sim_end_estimation - start_predict) > self.max_predict_time:
                        cancelled = True

        return value

    def simulate_sync(self, game, game_player, canonical_state_str):
        start_predict = time.time()
        for idx in range(self.numMCTSSims):
            start_sim_time = time.time()

            game_clone = game.clone()
            game_clone.reset_unknown_states(game_player)

            if idx == 0:
                # sanity check
                clone_canonical_state = game_clone.get_observation(game_player)
                clone_canonical_state_str = game_clone.get_observation_str(clone_canonical_state)

                if canonical_state_str != clone_canonical_state_str:
                    print("Warning: states are not equal after reset_unknown_states()")

            _, value = self.search_sync(game_clone, game_player)

            end_sim_time = time.time()

            if self.max_predict_time and idx != 0:
                next_sim_end_estimation = end_sim_time + (end_sim_time - start_sim_time)
                if (next_sim_end_estimation - start_predict) > self.max_predict_time:
                    break

        return value

    def predict_nnet_async(self, game, game_player):
        start_predict = time.time()
        pi, v = self.agent.predict(game, game_player)
        self.nnet_spent += time.time() - start_predict
        return pi, v

    def search_async(self, game, game_player):
        if game.is_ended():
            if game.is_draw():
                return True, -1
            return False, game.get_score(game_player)

        canonical_state = game.get_observation(game_player)
        canonical_state_str = game.get_observation_str(canonical_state)
        valid_actions = game.get_valid_moves(game_player)

        with self.node_lock[canonical_state_str]:
            if canonical_state_str not in self.Ps:
                self.simulations_num += 1

                future = self.predict_executor.submit(self.predict_nnet_async, game, game_player)
                result = future.result()

                self.Ps[canonical_state_str], value = result[0], result[1]
                self.Ps[canonical_state_str] = self.Ps[canonical_state_str] * valid_actions  # masking invalid moves
                self.Ps[canonical_state_str] /= np.sum(self.Ps[canonical_state_str])  # renormalize

                self.Ns[canonical_state_str] = 0

                return False, value

            cur_best = -float('inf')
            best_act = -1

            if sum(valid_actions) == 0:
                print("Sum of valid actions should not be 0!")
                return False, -1

            # pick the action with the highest upper confidence bound
            for action in range(game.get_action_size()):
                if valid_actions[action]:
                    if (canonical_state_str, action) in self.Qsa:
                        u = self.Qsa[(canonical_state_str, action)] \
                            + self.cpuct * self.Ps[canonical_state_str][action] * math.sqrt(
                            self.Ns[canonical_state_str]) / \
                              (1 + self.Nsa[(canonical_state_str, action)])
                    else:
                        u = self.cpuct * self.Ps[canonical_state_str][action] \
                            * math.sqrt(self.Ns[canonical_state_str] + EPS)  # Q = 0 ?

                    if u > cur_best:
                        cur_best = u
                        best_act = action

        action = best_act

        _, next_player = game.make_move(best_act)

        draw_result, value = self.search_async(game, next_player)

        if game_player != next_player and not draw_result:
            value = -value

        with self.node_lock[canonical_state_str]:
            if (canonical_state_str, action) in self.Qsa:
                self.Qsa[(canonical_state_str, action)] = (self.Nsa[(canonical_state_str, action)] * self.Qsa[
                    (canonical_state_str, action)] + value) / (self.Nsa[(canonical_state_str, action)] + 1)

                self.Nsa[(canonical_state_str, action)] += 1

            else:
                self.Qsa[(canonical_state_str, action)] = value
                self.Nsa[(canonical_state_str, action)] = 1

            self.Ns[canonical_state_str] += 1

        return False, value

    def search_sync(self, game, game_player):
        if game.is_ended():
            if game.is_draw():
                return True, -1
            return False, game.get_score(game_player)

        canonical_state = game.get_observation(game_player)
        canonical_state_str = game.get_observation_str(canonical_state)
        valid_actions = game.get_valid_moves(game_player)

        if canonical_state_str not in self.Ps:
            self.simulations_num += 1

            start_predict = time.time()

            self.Ps[canonical_state_str], value = self.agent.predict(game, game_player)

            self.nnet_spent += time.time() - start_predict

            self.Ps[canonical_state_str] = self.Ps[canonical_state_str] * valid_actions  # masking invalid moves
            self.Ps[canonical_state_str] /= np.sum(self.Ps[canonical_state_str])  # renormalize

            self.Ns[canonical_state_str] = 0

            return False, value

        cur_best = -float('inf')
        best_act = -1

        if sum(valid_actions) == 0:
            print("Sum of valid actions should not be 0!")
            return False, -1

        # pick the action with the highest upper confidence bound
        for action in range(game.get_action_size()):
            if valid_actions[action]:
                if (canonical_state_str, action) in self.Qsa:
                    u = self.Qsa[(canonical_state_str, action)] \
                        + self.cpuct * self.Ps[canonical_state_str][action] * math.sqrt(self.Ns[canonical_state_str]) / \
                          (1 + self.Nsa[(canonical_state_str, action)])
                else:
                    u = self.cpuct * self.Ps[canonical_state_str][action] \
                        * math.sqrt(self.Ns[canonical_state_str] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = action

        action = best_act

        _, next_player = game.make_move(action)

        draw_result, value = self.search_sync(game, next_player)

        if game_player != next_player and not draw_result:
            value = -value

        if (canonical_state_str, action) in self.Qsa:
            self.Qsa[(canonical_state_str, action)] = (self.Nsa[(canonical_state_str, action)] * self.Qsa[
                (canonical_state_str, action)] + value) / (self.Nsa[(canonical_state_str, action)] + 1)
            self.Nsa[(canonical_state_str, action)] += 1

        else:
            self.Qsa[(canonical_state_str, action)] = value
            self.Nsa[(canonical_state_str, action)] = 1

        self.Ns[canonical_state_str] += 1

        return False, value
