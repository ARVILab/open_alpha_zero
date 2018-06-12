import argparse
from EnvironmentSelector import EnvironmentSelector

from core.utils.utils import serialize, deserialize

from tqdm import *

import os
import sys
import math
import subprocess
import glob
from time import sleep

import tensorflow as tf
from keras import backend as K

from evaluator import AGENT_ACCEPTED_CODE, AGENT_REJECTED_CODE
from trainer import train
from self_play_generator import generate_self_play
from evaluator import evaluate
from collections import deque


def clean_dir(dir_path):
    if os.path.isdir(dir_path):
        files = glob.glob(dir_path + '/*')
        for f in files:
            os.remove(f)


def throw_error(message):
    print(message)
    sys.exit(1)


def fuse_memory(old_memory_path, new_memory_path, out_memory_path):
    if os.path.isfile(old_memory_path) and os.path.isfile(new_memory_path):
        try:
            serialize(deserialize(new_memory_path) + deserialize(old_memory_path), out_memory_path)
        except:
            print("Could not deserialize new + old. Try reverse order")
            serialize(deserialize(old_memory_path) + deserialize(new_memory_path), out_memory_path)
    elif os.path.isfile(new_memory_path):
        serialize(deserialize(new_memory_path), out_memory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")
    parser.add_argument("--workspace", dest="workspace",
                        help="Workspace of the training session. Required.")

    parser.add_argument("--memory_path", dest="memory_path",
                        help="Path to the game experience.")
    parser.add_argument("--agent_path", dest="agent_path",
                        help="Path to the agent's model.")

    parser.add_argument("--iterations", dest="iterations", default=100, type=int,
                        help="Number of iterations of Alpha Zero")

    parser.add_argument("--start_idx", dest="start_idx", default=0, type=int,
                        help="Index of iteration to start")

    parser.add_argument("--games_num", dest="games_num", default=100, type=int,
                        help="Number of games to play. ")

    parser.add_argument("--test_games_num", dest="test_games_num", type=int,
                        help="Number of games to play. ")

    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Show games outcome")
    parser.set_defaults(verbose=True)

    parser.add_argument('--debug', dest='debug', action='store_true', help="Show games per turn")
    parser.set_defaults(debug=False)

    parser.add_argument("--max_steps", dest="max_steps", type=int,
                        default=None,
                        help="Max steps in each game")

    parser.add_argument("--exploration_decay_steps", dest="exploration_decay_steps", type=int,
                        default=None,
                        help="Exploration decay in turns.")

    parser.add_argument('--skip_evaluation', dest='skip_evaluation', action='store_true',
                        help="Skip evaluation phase like in the latest version of Google's Alpha Zero.")
    parser.set_defaults(skip_evaluation=False)

    parser.add_argument('--optimize_for_inference', dest='optimize_for_inference', action='store_true',
                        help="Optimize for inference in self-play and evaluation phases")
    parser.set_defaults(optimize_for_inference=False)

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.workspace:
        parser.error('Workspace path must be selected')

    # define and create a workspace
    temp_dir = options.workspace + '/temp'
    temp_games_memory_dir = temp_dir + '/cluster_memory'

    self_play_temp_memory_path = temp_dir + '/self_play_memory.pkl'
    test_play_temp_memory_path = options.workspace + '/test_memory.pkl'

    os.makedirs(options.workspace, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(temp_games_memory_dir, exist_ok=True)

    # define primary agent model and primary memory model
    cur_agent_path = options.agent_path
    if not cur_agent_path:
        cur_agent_path = options.workspace + '/best'

    memory_path = options.workspace + '/memory.pkl'
    if options.memory_path is not None:
        if memory_path != options.memory_path:
            print("Synchronize memory in the target directory...")
            fuse_memory(memory_path, options.memory_path, memory_path)

        print("Loading memory...")
        self_play_examples_deque = deserialize(memory_path)
    else:
        self_play_examples_deque = deque([])

    # get Agent and Game instances by the profile
    env_selector = EnvironmentSelector()

    agent = env_selector.get_agent(options.agent_profile)
    agent_profile = env_selector.get_profile(options.agent_profile)
    game = env_selector.get_game(agent_profile.game)

    # set number of test games equal to the number of players in the game
    # this way every agent will play from different positions in the evaluation phase
    if not options.test_games_num:
        test_games_num = game.get_players_num()
    else:
        test_games_num = options.test_games_num

    # perform an initial training if agent's model was not specified
    if not options.agent_path:
        if options.memory_path:
            print("Agent model was not detected. Perform initial training...")
            agent.save(cur_agent_path)
            train(options.agent_profile, cur_agent_path, cur_agent_path,
                  game_memory=self_play_examples_deque, epochs=10)
        else:
            agent.save(cur_agent_path)
    else:
        agent.save(cur_agent_path)

    # main Alpha Zero loop

    loop_range = range(options.start_idx, options.iterations)

    if options.verbose:
        loop_range = tqdm(loop_range)

    for idx in loop_range:

        path_to_self_play_agent = cur_agent_path

        if options.optimize_for_inference:
            path_to_self_play_agent = cur_agent_path + ".pb"

            agent.load(cur_agent_path)
            agent.disable_training_capability(temp_dir=temp_dir, optimize=True)
            agent.save(path_to_self_play_agent)

        generate_self_play(options.agent_profile, path_to_self_play_agent, options.games_num,
                           memory_path, options.max_steps,
                           options.verbose, options.debug,
                           options.exploration_decay_steps, optimize_for_inference=False,
                           self_play_examples_deque=self_play_examples_deque)

        if options.optimize_for_inference:
            agent.enable_training_capability()

        contestant_agent_path = temp_dir + '/temp_contestant.h5'

        train(options.agent_profile, cur_agent_path, contestant_agent_path,
              game_memory=self_play_examples_deque, epochs=10)

        if options.skip_evaluation:

            agent.load(contestant_agent_path)
            agent.save(cur_agent_path)
            agent.save(options.workspace + '/model_updated_%d' % idx)

        else:

            updated = evaluate(options.agent_profile, contestant_agent_path, cur_agent_path,
                               test_games_num, experience_path=memory_path, acceptance_rate=0.6,
                               verbose=options.verbose, debug=True, max_steps=options.max_steps,
                               self_play_examples_deque=self_play_examples_deque)

            if updated:
                agent.load(contestant_agent_path)
                agent.save(cur_agent_path)
                agent.save(options.workspace + '/model_accepted_%d' % idx)
            else:
                agent.save(options.workspace + '/model_rejected_%d' % idx)
