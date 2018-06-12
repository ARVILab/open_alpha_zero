import argparse
from EnvironmentSelector import EnvironmentSelector
from core.utils.utils import serialize, deserialize
from core.World import World
from self_play_generator import generate_self_play

import subprocess
from time import sleep
import uuid
import GPUtil
import os

OPERATION_SUCCESSFUL = 0
INIT_CUDA_TIME_SEC = 10


def get_available_gpus():
    """ Get available GPU devices info. """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def start_process(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return process


def show_process_log(process):
    for line in iter(process.stdout.readline, ""):
        print(line)

    process.stdout.close()


def start_self_play_process(agent_profile, agent_path,
                            iteration_memory_path,
                            games_num, verbose, debug,
                            max_steps, exploration_decay_steps, num_gpu):
    # call trainer script
    # use temp dir to save all temp models and fuse them into iteration memory path

    command = "python3 self_play_generator.py"
    command += " --agent %s" % agent_profile
    command += " --agent_path %s" % agent_path
    command += " --out_experience_path %s" % iteration_memory_path
    command += " --games_num %d" % games_num
    command += " --max_steps %d" % max_steps
    if exploration_decay_steps:
        command += " --exploration_decay_steps %d" % exploration_decay_steps
    if verbose:
        command += " --verbose"
    if debug:
        command += " --debug"

    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % num_gpu

    return start_process(command)


def throw_error(message):
    print(message)
    sys.exit(1)


def generate_unique_memory_name():
    return "self_play_" + uuid.uuid4().hex[:6] + ".pkl"


if __name__ == "__main__":
    '''
    Uses self_play_generator.py to produce generated memory using all of the GPUs on the machine
    All individual memory collections are put into unique folders and when fused into one
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")
    parser.add_argument("--agent_path", dest="agent_path",
                        help="Path to the agent's model. Required.")
    parser.add_argument("--temp_path", dest="temp_path",
                        help="Path to the generated experience. Required.")
    parser.add_argument("--games_num", dest="games_num", type=int,
                        help="Number of games to play. Required.")

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

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.agent_path:
        parser.error('Agent path must be selected')

    if not options.temp_path:
        parser.error('Out experience path must be selected')

    if not options.games_num:
        parser.error('Number of games must be selected')

    num_gpus = len(GPUtil.getGPUs())

    if num_gpus < 1:
        throw_error("Host does not have GPU! Aborting...")

    if num_gpus == 1:
        print("Single-gpu machine detected, starting in the synchronous mode...")

        iteration_memory_path = options.temp_path + '/' + generate_unique_memory_name()

        generate_self_play(options.agent_profile, options.agent_path, options.games_num,
                           iteration_memory_path, options.max_steps,
                           options.verbose, options.debug, options.exploration_decay_steps)
    else:
        print("%d-gpu machine detected, starting in the asynchronous mode..." % num_gpus)

        processes = []

        for gpu_idx in range(num_gpus):
            if gpu_idx > 0:
                # let process take time to initialize CUDA. Ten seconds
                sleep(INIT_CUDA_TIME_SEC)

            iteration_memory_path = options.temp_path + '/' + generate_unique_memory_name()

            print("start self-play process on GPU %d" % gpu_idx)

            process = start_self_play_process(options.agent_profile, options.agent_path,
                                              iteration_memory_path,
                                              options.games_num, options.verbose,
                                              options.debug, options.max_steps, options.exploration_decay_steps,
                                              gpu_idx)

            processes.append(process)

        show_process_log(processes[num_gpus - 1])

        exit_codes = [process.wait() for process in processes]

        for idx, exit_code in enumerate(exit_codes):
            if exit_code != OPERATION_SUCCESSFUL:
                throw_error("Self-play process %d exited with error code %d!" % (idx, exit_code))
