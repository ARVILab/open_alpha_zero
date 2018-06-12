import argparse
from EnvironmentSelector import EnvironmentSelector
from core.utils.utils import serialize, deserialize
from core.World import World

import sys
from collections import deque


def generate_self_play(opt_agent_profile, agent_path, games_num,
                       experience_path, max_steps,
                       verbose, debug, exploration_decay_steps,
                       optimize_for_inference=False, self_play_examples_deque=deque([])):
    world = World()

    env_selector = EnvironmentSelector()

    agent = env_selector.get_agent(opt_agent_profile)

    agent.load(agent_path)

    agent_profile = env_selector.get_profile(opt_agent_profile)
    game = env_selector.get_game(agent_profile.game)

    if optimize_for_inference:
        agent.disable_training_capability()

    self_play_examples = world.generate_self_play(agent, game, games_num,
                                                  max_game_steps_n=max_steps,
                                                  verbose=verbose,
                                                  show_every_turn=debug,
                                                  exploration_decay_steps=exploration_decay_steps)

    self_play_examples_deque += self_play_examples

    serialize(self_play_examples_deque, experience_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")
    parser.add_argument("--agent_path", dest="agent_path",
                        help="Path to the agent's model. Required.")
    parser.add_argument("--out_experience_path", dest="experience_path",
                        help="Path to the generated experience. Required.")
    parser.add_argument("--games_num", dest="games_num", type=int,
                        help="Number of games to play. Required.")

    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Show games outcome")
    parser.set_defaults(verbose=True)

    parser.add_argument('--debug', dest='debug', action='store_true', help="Show games per turn")
    parser.set_defaults(verbose=False)

    parser.add_argument("--max_steps", dest="max_steps", type=int,
                        default=None,
                        help="Max steps in each game")

    parser.add_argument("--exploration_decay_steps", dest="exploration_decay_steps", type=int,
                        default=None,
                        help="Exploration decay in turns.")

    parser.add_argument('--optimize_for_inference', dest='optimize_for_inference', action='store_true',
                        help="Optimize for inference in self-play and evaluation phases")
    parser.set_defaults(optimize_for_inference=False)

    parser.add_argument("--temp_path", dest="temp_path",
                        help="Path to the generated experience. Required.")

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.agent_path:
        parser.error('Agent path must be selected')

    if not options.experience_path:
        parser.error('Out experience path must be selected')

    if not options.games_num:
        parser.error('Number of games must be selected')

    generate_self_play(options.agent_profile, options.agent_path, options.games_num,
                       options.experience_path, options.max_steps,
                       options.verbose, options.debug, options.exploration_decay_steps)
