import numpy as np
from tqdm import *
import itertools


# TODO test games with checkers, when preferans, when update comments in Game
class World():
    """
    An World class where any agents can be play and generate experience
    """

    def __init__(self):
        self.RESULT_DRAW = -1

    def execute_game(self, agents, game, max_game_steps_n=None, allow_exploration=False,
                     verbose=False, show_every_turn=False, exploration_decay_steps=None, need_reset=True):

        episode_exp = []
        augmented_exp = []

        if need_reset:
            game.reset()

        game_results = []
        for idx, agent in enumerate(agents):
            if allow_exploration:
                agent.set_exploration_enabled(True)
            else:
                agent.set_exploration_enabled(False)
            agent.prepare_to_game()
            game_results.append(self.RESULT_DRAW)

        cur_player = game.get_cur_player()

        loop_range = itertools.count()

        if verbose:
            loop_range = tqdm(loop_range)

        for episodeStep in loop_range:
            if max_game_steps_n is not None and episodeStep > max_game_steps_n:
                episode_exp = []
                break

            if show_every_turn:
                print("\n", game.get_display_str())

            observation = game.get_observation(cur_player)

            cur_turn_agent = agents[cur_player]

            if exploration_decay_steps and episodeStep >= exploration_decay_steps:
                cur_turn_agent.set_exploration_enabled(False)

            actions_prob, observation_value = cur_turn_agent.predict(game, cur_player)

            episode_exp.append([observation, cur_player, actions_prob])

            if not allow_exploration:
                bestA = np.argmax(actions_prob)
                actions_prob = [0] * len(actions_prob)
                actions_prob[bestA] = 1

            action = np.random.choice(len(actions_prob), p=actions_prob)

            _, cur_player = game.make_move(action)

            cur_turn_agent.on_turn_finished(game)

            if game.is_ended():
                if not game.is_draw():
                    print("\n")
                    for idx, agent in enumerate(agents):
                        game_results[idx] = game.get_score(idx)
                        if verbose:
                            print(agent.get_name(), " scored ", game_results[idx])
                    print("\n")
                else:
                    for idx, agent in enumerate(agents):
                        game_results[idx] = self.RESULT_DRAW
                break

        if verbose:
            print("\n\nFinal observation on step %d.\n%s\n" % (
                episodeStep, game.get_display_str()))

        for idx, [cur_observation, cur_player, cur_pi] in enumerate(episode_exp):
            augmented_exp.append((cur_observation, cur_pi, game_results[cur_player]))

        return augmented_exp, game_results

    def execute_games(self, agents, game, num_games, max_game_steps_n=None, allow_exploration=False,
                      verbose=False, show_every_turn=False, exploration_decay_steps=None):
        games_experience = []
        games_results = [0] * len(agents)

        for idx in range(len(agents)):
            games_results[idx] = 0

        loop_range = range(num_games)

        if verbose:
            loop_range = tqdm(loop_range)

        for idx in loop_range:
            game_experience, game_results = self.execute_game(agents, game,
                                                              max_game_steps_n=max_game_steps_n,
                                                              allow_exploration=allow_exploration, verbose=verbose,
                                                              show_every_turn=show_every_turn,
                                                              exploration_decay_steps=exploration_decay_steps)

            if len(game_experience) > 0:
                games_experience.extend(game_experience)

            for idx in range(len(agents)):
                games_results[idx] += game_results[idx]

        if verbose:
            for idx, agent in enumerate(agents):
                print("%s: %d" % (agent.get_name(), games_results[idx]))

        return games_experience, games_results

    def generate_self_play(self, agent, game, num_games,
                           max_game_steps_n=None, allow_exploration=True,
                           verbose=False, show_every_turn=False,
                           exploration_decay_steps=None):
        agents = []
        for idx in range(game.get_players_num()):
            agents.append(agent.clone())

        games_experience, _ = self.execute_games(agents, game, num_games,
                                                 max_game_steps_n=max_game_steps_n, allow_exploration=allow_exploration,
                                                 verbose=verbose, show_every_turn=show_every_turn,
                                                 exploration_decay_steps=exploration_decay_steps)
        return games_experience
