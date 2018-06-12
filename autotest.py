import argparse
import subprocess
import os
import os.path
import shutil

OPERATION_SUCCESSFUL = 0


def execute_commant_synch(command, show_output=True):
    print("execute command: ", command)

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, universal_newlines=True)

    if show_output:
        for line in iter(process.stdout.readline, ""):
            print(line)

    process.stdout.close()
    process.wait()
    return process.returncode


def run_test(test_name, test_command):
    result_code = execute_commant_synch(test_command)

    assert result_code == OPERATION_SUCCESSFUL, "Test: %s has not passed." % test_name
    print("Test: %s has passed." % test_name)


def stop_testing(options):
    shutil.rmtree(options.workspace)
    print("All tests passed!")
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")

    parser.add_argument("--workspace", dest="workspace",
                        help="Workspace of the training session. Required.")

    parser.add_argument("--hosts", dest="hosts",
                        default=None,
                        help="Hosts to run Alpha Zero. The folder of the repo must be shared between all of them")

    parser.add_argument('--check_sync_training', dest='check_sync_training', action='store_true',
                        help="Check sync training.")
    parser.set_defaults(check_sync_training=False)

    parser.add_argument('--check_cluster_training', dest='check_cluster_training', action='store_true',
                        help="Check Horovod cluster learning. Hosts option required.")
    parser.set_defaults(check_cluster_training=False)

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.workspace:
        parser.error('Workspace path must be selected')

    if os.path.isdir(options.workspace):
        shutil.rmtree(options.workspace)

    if not options.check_sync_training:
        exec_command = "python3 alpha_zero_trainer.py"
    else:
        exec_command = "python3 alpha_zero_trainer_sync.py"

    #
    # Basic training
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100

    run_test("Basic training", command)

    #
    # Check memory generation
    #

    if not os.path.isfile(options.workspace + "/memory.pkl"):
        assert False, "Previous iterations of training should have produced a memory file!"

    #
    # Training from memory
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --memory_path %s" % options.workspace + "/memory.pkl"
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100

    run_test("Training from memory", command)

    #
    # Check model generation
    #

    if os.path.isfile(options.workspace + "/best.h5"):
        model_name = "/best.h5"
    elif os.path.isfile(options.workspace + "/model_updated_0.h5"):
        model_name = "/model_updated_0.h5"
    elif os.path.isfile(options.workspace + "/model_rejected_0.h5"):
        model_name = "/model_rejected_0.h5"
    else:
        assert False, "Previous iterations of training should have produced a model!"

    #
    # Training from checkpoint
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --agent_path %s" % options.workspace + model_name
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100

    run_test("Training from the checkpoint", command)

    #
    # Training from checkpoint and memory
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --agent_path %s" % options.workspace + model_name
    command += " --memory_path %s" % options.workspace + "/memory.pkl"
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100

    run_test("Training from the checkpoint and memory", command)

    #
    # Training from checkpoint and memory with inference optimization
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --agent_path %s" % options.workspace + model_name
    command += " --memory_path %s" % options.workspace + "/memory.pkl"
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100
    command += " --optimize_for_inference"

    run_test("Training from the checkpoint and memory with inference optimization", command)

    if not options.hosts:
        stop_testing(options)

    #
    # Training from checkpoint and memory with native distributed option
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --agent_path %s" % options.workspace + model_name
    command += " --memory_path %s" % options.workspace + "/memory.pkl"
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100
    command += " --train_distributed_native"
    command += " --hosts %s" % options.hosts

    run_test("Training from the checkpoint and memory with native distributed option", command)

    if not options.check_cluster_training:
        stop_testing(options)

    #
    # Training from checkpoint and memory with native distributed option
    #

    command = exec_command
    command += " --agent %s" % options.agent_profile
    command += " --agent_path %s" % options.workspace + model_name
    command += " --memory_path %s" % options.workspace + "/memory.pkl"
    command += " --workspace %s" % options.workspace
    command += " --games_num %d" % 1
    command += " --iterations %d" % 1
    command += " --max_steps %d" % 100
    command += " --train_distributed"
    command += " --hosts %s" % options.hosts

    run_test("Training from the checkpoint and memory with Horovod distributed mode", command)

    stop_testing(options)
