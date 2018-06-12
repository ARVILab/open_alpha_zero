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

OPERATION_SUCCESSFUL = 0

SYNCH_MAX_CHECK_NUM = 5 * 60  # wait for synchronization max 10 minutes
SYNCH_CHECKS_REMAINING = SYNCH_MAX_CHECK_NUM
SYNCH_CHECK_TIME_SEC = 1


class Host():
    def __init__(self, ip, num_gpu):
        self.ip = ip
        self.num_gpu = num_gpu


def parse_host(str):
    parts = str.split(":")

    if len(parts) != 2:
        throw_error("Invalid host: %s" % str)

    return Host(parts[0], int(parts[1]))


def generate_open_mpi_distributed_command(hosts, use_gpu):
    processes = 0
    hosts_string = ""
    for idx, host in enumerate(hosts):
        if use_gpu:
            processes += host.num_gpu
            hosts_string += ("," if idx > 0 else "") + "%s:%d" % (host.ip, host.num_gpu)
        else:
            processes += 1
            hosts_string += ("," if idx > 0 else "") + host.ip

    command = "mpirun"
    command += " -np %d" % processes
    command += " -H %s" % hosts_string

    command += " -bind-to none"
    command += " -map-by slot"
    command += " -x NCCL_DEBUG=INFO"
    command += " -x LD_LIBRARY_PATH"
    command += " -x PATH"
    command += " -mca pml ob1 -mca btl ^openib"

    return command


def execute_commant_synch(command, show_output=True):
    print("execute command: ", command)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

    if show_output:
        for line in iter(process.stdout.readline, ""):
            print(line)

    process.stdout.close()
    process.wait()
    return process.returncode


def train(agent_profile, memory_path, cur_agent_path, new_agent_path,
          hosts=None, train_distributed=False, train_distributed_native=False,
          epochs=1):
    # call trainer script
    command = "python3 trainer.py"
    command += " --agent %s" % agent_profile
    command += " --memory_path %s" % memory_path
    command += " --out_agent_path %s" % new_agent_path
    command += " --epochs %s" % epochs
    if cur_agent_path is not None:
        command += " --agent_path %s" % cur_agent_path

    if hosts is not None:
        if len(hosts) == 0:
            throw_error("At least one host should be specified!")

        if train_distributed_native:
            command += " --train_distributed_native"
        else:
            if train_distributed:
                # train in cluster. Requires at least 25 gb/s network bandwidth
                command += " --train_distributed"
                open_mpi_command = generate_open_mpi_distributed_command(hosts, True)
            else:
                # train on the main server
                main_server = Host("localhost", hosts[0].num_gpu)
                open_mpi_command = generate_open_mpi_distributed_command([main_server], True)

            command = open_mpi_command + " " + command

    code = execute_commant_synch(command)

    print("training finished, exit code: ", code)

    if code != OPERATION_SUCCESSFUL:
        throw_error("Could not perform agent training!")


def generate_self_play(agent_profile, agent_path, temp_dir, iteration_memory_path,
                       games_num, verbose, debug, max_steps, exploration_decay_steps,
                       hosts=None):
    # call trainer script
    # use temp dir to save all temp models and fuse them into iteration memory path

    clean_dir(temp_dir)
    if os.path.isfile(iteration_memory_path):
        os.remove(iteration_memory_path)

    distributed = hosts is not None

    if not distributed:
        command = "python3 self_play_generator.py"
        command += " --agent %s" % agent_profile
        command += " --agent_path %s" % agent_path
        command += " --out_experience_path %s" % iteration_memory_path
        command += " --games_num %d" % games_num
        if max_steps:
            command += " --max_steps %d" % max_steps
        if exploration_decay_steps:
            command += " --exploration_decay_steps %d" % exploration_decay_steps
        if verbose:
            command += " --verbose"
        if debug:
            command += " --debug"
    else:
        processes = 0
        for host in hosts:
            processes += host.num_gpu

        games_num = math.ceil(games_num / processes)

        command = "python3 self_play_generator_distributed.py"
        command += " --agent %s" % agent_profile
        command += " --agent_path %s" % agent_path
        command += " --temp_path %s" % temp_dir
        command += " --games_num %d" % games_num
        if max_steps:
            command += " --max_steps %d" % max_steps
        if exploration_decay_steps:
            command += " --exploration_decay_steps %d" % exploration_decay_steps
        if verbose:
            command += " --verbose"
        if debug:
            command += " --debug"

        open_mpi_command = generate_open_mpi_distributed_command(hosts, False)

        command = open_mpi_command + " " + command

    code = execute_commant_synch(command)

    print("self-play generation finished, exit code: ", code)

    if distributed:

        processes = 0
        for idx, host in enumerate(hosts):
            processes += host.num_gpu

        print("Self-play memory collector started...")

        files = glob.glob(temp_dir + '/*.pkl')

        print("Detected %d memory pieces" % len(files))
        SYNCH_CHECKS_REMAINING = SYNCH_MAX_CHECK_NUM
        while SYNCH_CHECKS_REMAINING >= 0:
            if len(files) < processes:
                print("Warning: memory files are not synchronized yet: %d/%d" % (len(files), processes))
                sleep(SYNCH_CHECK_TIME_SEC)
                SYNCH_CHECKS_REMAINING -= SYNCH_CHECK_TIME_SEC
            else:
                break

        for memory_temp_file in files:
            print("loading memory file ", memory_temp_file)
            fuse_memory(iteration_memory_path, memory_temp_file, iteration_memory_path)

        print("Memory collection finished.")

    if code != OPERATION_SUCCESSFUL:
        throw_error("Could not perform self-play generation!")


def evaluate(agent_profile, contestant_agent_path,
             cur_agent_path, test_memory_path,
             test_games_num, verbose, debug, max_steps):
    # call the evaluator script
    # return boolean using return code of the script

    if os.path.isfile(test_memory_path):
        os.remove(test_memory_path)

    command = "python3 evaluator.py"
    command += " --agent %s" % agent_profile
    command += " --agent_new_path %s" % contestant_agent_path
    command += " --agent_old_path %s" % cur_agent_path
    command += " --games_num %d" % test_games_num
    command += " --out_experience_path %s" % test_memory_path
    if max_steps:
        command += " --max_steps %d" % max_steps
    if verbose:
        command += " --verbose"
    if debug:
        command += " --debug"

    code = execute_commant_synch(command)

    print("model evaluation finished, exit code: ", code)

    if code != AGENT_ACCEPTED_CODE and code != AGENT_REJECTED_CODE:
        throw_error("Could not perform agent evaluation!")

    return code == AGENT_ACCEPTED_CODE


def fuse_memory(old_memory_path, new_memory_path, out_memory_path):
    if os.path.isfile(old_memory_path) and os.path.isfile(new_memory_path):
        try:
            serialize(deserialize(new_memory_path) + deserialize(old_memory_path), out_memory_path)
        except:
            print("Could not deserialize new + old. Try reverse order")
            serialize(deserialize(old_memory_path) + deserialize(new_memory_path), out_memory_path)
    elif os.path.isfile(new_memory_path):
        serialize(deserialize(new_memory_path), out_memory_path)


def clean_dir(dir_path):
    if os.path.isdir(dir_path):
        files = glob.glob(dir_path + '/*')
        for f in files:
            os.remove(f)


def throw_error(message):
    print(message)
    sys.exit(1)


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

    parser.add_argument("--hosts", dest="hosts",
                        default=None,
                        help="Hosts to run Alpha Zero. The folder of the repo must be shared between all of them")

    parser.add_argument('--train_distributed', dest='train_distributed', action='store_true',
                        help="Train NN in cluster specified by hosts option")
    parser.set_defaults(train_distributed=False)

    parser.add_argument('--skip_evaluation', dest='skip_evaluation', action='store_true',
                        help="Skip evaluation phase like in the latest version of Google's Alpha Zero.")
    parser.set_defaults(skip_evaluation=False)

    parser.add_argument('--train_distributed_native', dest='train_distributed_native', action='store_true',
                        help="Enable native distributed training on main machine")
    parser.set_defaults(train_distributed_native=False)

    parser.add_argument('--optimize_for_inference', dest='optimize_for_inference', action='store_true',
                        help="Optimize for inference in self-play and evaluation phases")
    parser.set_defaults(optimize_for_inference=False)

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.workspace:
        parser.error('Workspace path must be selected')

    # parse hosts
    hosts = None
    if options.hosts:
        hosts = []
        hosts_strs = options.hosts.split(",")
        for host_str in hosts_strs:
            hosts.append(parse_host(host_str))

    # force agent to use CPU in the main script
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
    session = tf.Session(config=config)
    K.set_session(session)

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
            train(options.agent_profile, options.memory_path, None, cur_agent_path,
                  hosts=hosts, train_distributed=options.train_distributed,
                  train_distributed_native=options.train_distributed_native,
                  epochs=10)
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

        generate_self_play(options.agent_profile, path_to_self_play_agent,
                           temp_games_memory_dir, self_play_temp_memory_path,
                           options.games_num, options.verbose,
                           options.debug, options.max_steps, options.exploration_decay_steps,
                           hosts=hosts)

        if options.optimize_for_inference:
            agent.enable_training_capability()

        fuse_memory(memory_path, self_play_temp_memory_path, memory_path)

        contestant_agent_path = temp_dir + '/temp_contestant.h5'

        train(options.agent_profile, memory_path, cur_agent_path, contestant_agent_path,
              hosts=hosts, train_distributed=options.train_distributed,
              train_distributed_native=options.train_distributed_native,
              epochs=1)

        if options.skip_evaluation:

            agent.load(contestant_agent_path)
            agent.save(cur_agent_path)
            agent.save(options.workspace + '/model_updated_%d' % idx)

        else:

            updated = evaluate(options.agent_profile, contestant_agent_path,
                               cur_agent_path, test_play_temp_memory_path,
                               test_games_num, options.verbose,
                               True, options.max_steps)

            fuse_memory(memory_path, test_play_temp_memory_path, memory_path)

            if updated:
                agent.load(contestant_agent_path)
                agent.save(cur_agent_path)
                agent.save(options.workspace + '/model_accepted_%d' % idx)
            else:
                agent.save(options.workspace + '/model_rejected_%d' % idx)
