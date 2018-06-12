from abc import ABC, abstractclassmethod


class Game(ABC):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. 
    
    This class supports games with >0 players.
    This class supports rewards in range [-1, 1]
    This class assumes that DRAW result is -1 (loss for both players)
    This class supports player indexes from 0, 1, 2 etc
    
    see checkers/CheckersGame.py for an example implementation
    """

    def __init__(self):
        pass

    @abstractclassmethod
    def reset(self):
        """
            Resets game to the initial state
        """
        pass

    @abstractclassmethod
    def get_cur_player(self):
        """
        Returns:
            int: current player idx
        """
        pass

    @abstractclassmethod
    def get_players_num(self):
        """
        Returns:
            int: number of players
        """
        pass

    @abstractclassmethod
    def get_action_size(self):
        """
        Returns:
            int: number of all possible actions
        """
        pass

    @abstractclassmethod
    def get_observation_size(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    @abstractclassmethod
    def make_move(self, action):
        """
        Input:
            action: action taken by the current player

        Returns:
            double: score of current player on the current turn
            int: player who plays in the next turn
        """
        pass

    @abstractclassmethod
    def get_valid_moves(self, player):
        """
        Input:
            player: player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    @abstractclassmethod
    def is_ended(self):
        """
        This method must return True if is_draw_situation returns True
        Returns:
            boolean: False if game has not ended. True otherwise
        """
        pass

    @abstractclassmethod
    def is_draw(self):
        """
        Returns:
            boolean: True if game ended in a draw, False otherwise
        """
        pass

    @abstractclassmethod
    def get_score(self, player):
        """
        Input:
            player: current player

        Returns:
            double: reward in [-1, 1] for player if game has ended
        """
        pass

    @abstractclassmethod
    def get_observation(self, player):
        """
        Input:
            player: current player

        Returns:
            observation matrix which will serve as an input to agent.predict
        """
        pass

    @abstractclassmethod
    def get_observation_str(self, observation):
        """
        Input:
            canonical_state: observation

        Returns:
            string: a quick conversion of state to a string format.
                    Required by MCTS for hashing.
        """
        pass

    @abstractclassmethod
    def get_display_str(self):
        """
        Returns:
            string: a display of current game state
        """
        pass

    @abstractclassmethod
    def clone(self):
        """
        Returns:
            Game: a deep clone of current Game object
        """
        pass

    @abstractclassmethod
    def reset_unknown_states(self, player):
        """
        Resets unknown inner states of objects outside of observation of the player. 
        For example, in cards-based games this function might reshuffle current state of opponent deck
        """
        pass
