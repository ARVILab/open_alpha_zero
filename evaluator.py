import argparse
from EnvironmentSelector import EnvironmentSelector
from core.agents.AgentMCTS import AgentMCTS
from core.utils.utils import serialize, deserialize
from core.World import World

from collections import deque

import sys

AGENT_ACCEPTED_CODE = 0
AGENT_REJECTED_CODE = 1


def shift_list(list_obj, n):
    if n == 0:
        return list_obj
    return list_obj[n:] + list_obj[:n]


def evaluate(agent_profile, agent_new_path, agent_old_path,
             games_num, experience_path=None, acceptance_rate=0.6,
             verbose=True, debug=False, max_steps=None, self_play_examples_deque=deque([])):

    env_selector = EnvironmentSelector()
    agent = env_selector.get_agent(agent_profile)
    agent.set_exploration_enabled(False)

    agent_profile = env_selector.get_profile(agent_profile)
    game = env_selector.get_game(agent_profile.game)

    agents = []

    for idx in range(game.get_players_num()):
        old_agent = agent.clone()
        old_agent.load(agent_old_path)
        agents.append(old_agent)

    agent.load(agent_new_path)

    agents[0] = agent

    arena_games_results = [0] * len(agents)
    arena_examples = []
    arena_games_n = int(games_num / game.get_players_num())

    world = World()

    for jdx in range(game.get_players_num()):
        playing_agents = shift_list(agents, jdx)

        sess_arena_examples, games_results = world.execute_games(playing_agents,
                                                                 game,
                                                                 arena_games_n,
                                                                 max_game_steps_n=max_steps,
                                                                 verbose=verbose,
                                                                 show_every_turn=debug)

        games_results = shift_list(games_results, -jdx)

        for index in range(len(arena_games_results)):
            arena_games_results[index] += games_results[index]

        arena_examples.extend(sess_arena_examples)

    self_play_examples_deque += arena_examples

    if experience_path:
        serialize(self_play_examples_deque, experience_path)

    cur_rewards = arena_games_results[0]
    other_rewards = sum(arena_games_results) - cur_rewards

    if verbose:
        print("Current agent got rewards: %d\n"
              "Total reward across all other agents: %d" % (cur_rewards, other_rewards))

    updated = (cur_rewards > other_rewards) >= acceptance_rate

    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent", dest="agent_profile",
                        help="Agent profile from EnvironmentSelector. Required.")

    parser.add_argument("--agent_new_path", dest="agent_new_path",
                        help="Path to the new agent's model. Required.")

    parser.add_argument("--agent_old_path", dest="agent_old_path",
                        help="Path to the old agent's model. Required.")

    parser.add_argument("--games_num", dest="games_num", type=int,
                        help="Number of games to play. Required.")

    parser.add_argument("--out_experience_path", dest="experience_path", default=None,
                        help="Path to the generated experience.")

    parser.add_argument("--acceptance_rate", dest="acceptance_rate", default=0.6, type=float,
                        help="Acceptance rate")

    parser.add_argument('--verbose', dest='verbose', action='store_true', help="Show games outcome")
    parser.set_defaults(verbose=True)

    parser.add_argument('--debug', dest='debug', action='store_true', help="Show games per turn")
    parser.set_defaults(debug=False)

    parser.add_argument("--max_steps", dest="max_steps", type=int,
                        default=None,
                        help="Max steps in each game")

    options = parser.parse_args()

    if not options.agent_profile:
        parser.error('Agent profile must be selected')

    if not options.agent_new_path:
        parser.error('New Agent path must be selected')

    if not options.agent_old_path:
        parser.error('Old Agent path must be selected')

    if not options.games_num:
        parser.error('Number of games must be selected')

    updated = evaluate(options.agent_profile, options.agent_new_path, options.agent_old_path,
                       options.games_num, experience_path=options.experience_path,
                       acceptance_rate=options.acceptance_rate,
                       verbose=options.verbose, debug=options.debug, max_steps=options.max_steps)

    if updated:
        sys.exit(AGENT_ACCEPTED_CODE)
    else:
        sys.exit(AGENT_REJECTED_CODE)
