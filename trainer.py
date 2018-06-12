import argparse
from EnvironmentSelector import EnvironmentSelector

from core.utils.utils import serialize, deserialize

import sys


def throw_error(message):
    print(message)
    sys.exit(1)


def train(agent_profile, agent_path, out_agent_path,
          memory_path=None, game_memory=None,
          train_distributed=False, train_distributed_native=False,
          epochs=1):
    env_selector = EnvironmentSelector()

    agent = env_selector.get_agent(agent_profile, native_multi_gpu_enabled=train_distributed_native)

    if agent_path:
        agent.load(agent_path)

    if not game_memory:
        if not memory_path:
            print("Error: You must specify either game memory or memory path!")
            throw_error("Error: You must specify either game memory or memory path!")

        print("deserializing memory from the memory model...")

        game_memory = deserialize(memory_path)

    print("%d steps loaded from memory" % len(game_memory))

    print("Initiate training...")

    agent.train(game_memory, epochs=epochs)

    print("Training finished!")

    if train_distributed:
        import horovod.tensorflow as hvd
        if hvd.rank() == 0:
            # save only on the main server
            agent.save(out_agent_path)
    else:
        agent.save(out_agent_path)

    print("Model saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")
    parser.add_argument("--memory_path", dest="memory_path",
                        help="Agent profile from EnvironmentSelector. Required.")
    parser.add_argument("--out_agent_path", dest="out_agent_path",
                        help="Path to the generated agent's model. Required.")

    parser.add_argument("--agent_path", dest="agent_path",
                        help="Path to the agent's model.")

    parser.add_argument('--train_distributed', dest='train_distributed', action='store_true',
                        help="Train NN in cluster specified by hosts option")
    parser.set_defaults(train_distributed=False)

    parser.add_argument('--train_distributed_native', dest='train_distributed_native', action='store_true',
                        help="Enable native distributed training on main machine")
    parser.set_defaults(train_distributed_native=False)

    parser.add_argument("--epochs", dest="epochs", type=int,
                        default=1,
                        help="Epochs to train")

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.memory_path:
        parser.error('Memory path must be selected')

    if not options.out_agent_path:
        parser.error('Out Agent model path must be selected')

    train(options.agent_profile, options.agent_path,
          options.out_agent_path, memory_path=options.memory_path,
          train_distributed=options.train_distributed,
          train_distributed_native=options.train_distributed_native,
          epochs=options.epochs)
