# Building your own Alpha Zero. Part 2: Decisions

*"I've looked forward in time to see all of the possible outcomes of a coming conflict." - Doctor Stephen Vincent Strange, a superhero from Marvel Cinematic Universe.* 

If only we could predict all of the outcomes of our actions ahead of time! Unfortunately we can't and neither do the machines, unless a task they are triyng to solve is a really simple one. However, the machines still can predict future states of the game a lot faster than any human. It lets AI outperform humans in checkers, chess, go and various other games. Alpha Zero utilizes the power of predictions as well using an algorithm called **Monte Carlo Tree Search (MCTS)**.
This algorithm lets an AI to look into the most promising of the possible futures, instead of just trying to imagine every possible situation. Alpha Zero combines **MCTS** with a deep neural network, capable of evaluation of a game state, to determine these most promising futures. Given enough training data, this approach gives Alpha Zero the ability to efficiently find a best possible way to victory from any game state.

The best way to show learn how the MCTS works is to implement it. But, before we start our implementations, let's do some basic coding first. 

# Games and Agents

We are going to define two simple interfaces in Python: a **Game** and an **Agent**. The Game interface let's us create a game session, get observations of a current state, take actions and get result of a game session. The Agent interface let's us predict a policy and a value from a current game state.  

```
class Game():
    @abstractclassmethod
    def get_cur_player(self):
        """
        Returns:
            int: current player idx
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
    def get_valid_actions(self, player):
        """
        Input:
            player: player

        Returns:
            validActions: a binary vector of length self.get_action_size(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass
    
    @abstractclassmethod
    def take_action(self, action):
        """
        Input:
            action: action taken by the current player

        Returns:
            double: score of current player on the current turn
            int: player who plays in the next turn
        """
        pass
    
    @abstractclassmethod
    def get_observation_size(self):
        """
        Returns:
            (x,y,z): a tuple of observation dimensions
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
            observation: observation

        Returns:
            string: a quick conversion of state to a string format.
                    Required by MCTS for hashing.
        """
        pass
    
    @abstractclassmethod
    def is_ended(self):
        """
        This method must return True if is_draw returns True
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
    def clone(self):
        """
        Returns:
            Game: a deep clone of current Game object
        """
        pass
```

```
class Agent():

        @abstractclassmethod
        def predict(self, game, game_player):
            """
            Returns:
                policy, value: stochastic policy and a continuous value of a game observation 
            """
            pass
```

Take a minute or two to get familiar with these interfaces. When you are ready, we are going right into the implementation of the MCTS!

# MCTS-based Agent

Now we are going to make a MCTS-based Agent. I know, it's weird to implement an algorithm without explanations, but trust me on this one - it's not as hard as it looks. Reading the implementation will give you a basic idea of how the algorithm works and when I'll explain it all in a more formal way.  
The idea of an implementation was taken from the github repo [Alpha Zero General](https://github.com/suragnair/alpha-zero-general): 

```
class AgentMCTS():
        
        NO_EXPLORATION = 0
        EXPLORATION_TEMP = 1
        
        def __init__(self, agent, exp_rate=EXPLORATION_TEMP, cpuct=1, num_simulations=10):
            self.agent = agent
            self.cpuct = cpuct
            self.exp_rate = exp_rate
            self.num_simulations = num_simulations
            self.Qsa = {}  # stores Q values for s,a
            self.Nsa = {}  # stores times edge s,a was visited
            self.Ns = {}  # stores times observation s was visited
            self.Ps = {}  # stores initial policy (returned by neural net)

        def predict(self, game, game_player):
            """
            Returns:
                pi, value: stochastic policy and a continuous value of a game observation 
            """
            observation = game.get_observation(game_player)
            observation_str = game.get_observation_str(observation)
            
            for i in range(self.num_simulations):
                game_clone = game.clone()
                _, value = self.search(game_clone, game_player)
            
            counts = [self.Nsa[(observation_str, a)] if (observation_str, a) in self.Nsa else 0 for a in
                  range(game.get_action_size())]

            if self.exp_rate == AgentMCTS.NO_EXPLORATION:
                bestA = np.argmax(counts)
                policy = [0] * len(counts)
                policy[bestA] = 1
            else:
                counts = [x ** (1. / self.exp_rate) for x in counts]
                policy = [x / float(sum(counts)) for x in counts]
    
            return policy, value
            
        def search(self, game, game_player, first_search=False):
            """
            Returns:
                is_draw, value: a draw flag and a continuous value of a game observation 
            """
            
            # check for a terminal state
            if game.is_ended():
                if game.is_draw():
                    return True, -1
                return False, game.get_score(game_player)
        
            observation = game.get_observation(game_player)
            observation_str = game.get_observation_str(observation)
            valid_actions = game.get_valid_actions(game_player)

            # check if this observation is a previously unknown state
            if (observation_str not in self.Ps):
                # get an initial estimation of a policy and a value with a neural network-based agent
                self.Ps[observation_str], value = self.agent.predict(game, game_player)
                self.Ps[observation_str] = self.Ps[observation_str] * valid_actions  # masking invalid moves
                self.Ps[observation_str] /= np.sum(self.Ps[observation_str])  # renormalize
                self.Ns[observation_str] = 0
                return False, value

            cur_best = -float('inf')
            best_act = -1

            # pick the action with the highest upper confidence bound
            for action in range(game.get_action_size()):
                if valid_actions[action]:
                    if (observation_str, action) in self.Qsa:
                        u = self.Qsa[(observation_str, action)] \
                            + self.cpuct * self.Ps[observation_str][action] * math.sqrt(self.Ns[observation_str]) / \
                                 (1 + self.Nsa[(observation_str, action)])
                    else:
                        u = self.cpuct * self.Ps[observation_str][action] * math.sqrt(self.Ns[observation_str] + EPS)
        
                    if u > cur_best:
                        cur_best = u
                        best_act = action

            # take the best action
            _, next_player = game.take_action(best_act)
            draw_result, value = self.search(game, next_player)

            # update values after search is done
            if game_player != next_player and not draw_result:
                value = -value

            if (observation_str, action) in self.Qsa:
                self.Qsa[(observation_str, action)] = (self.Nsa[(observation_str, action)] * self.Qsa[
                        (observation_str, action)] + value) / (self.Nsa[(observation_str, action)] + 1)
                self.Nsa[(observation_str, action)] += 1
            else:
                self.Qsa[(observation_str, action)] = value
                self.Nsa[(observation_str, action)] = 1
                    
            self.Ns[observation_str] += 1
            return False, value
```

Woah! That was a lot of code to process. It's about time to explain what is going on here.

# Monte Carlo Tree Search

MCTS is a policy search algorithm that balances exploration with exploitation to output an improved policy after a number of simulations of the game. MCTS builds a tree where nodes are different observations and a directed edge exists between two nodes if a valid action can cause state to transition from one node to another. 
For each edge, we maintain a **Q** value denoted by **Q(s, a)** which is the expected reward for taking that action and **N(s, a)** which represents the number of times we took action a from state s across different simulations.

The hyperparametres of MCTS are:
* **number of simulations** is the parameter which represents how many previously unexplored nodes we want to visit every time we call *agent.predict* function;
* **cpuct** is the parameter controlling the degree of exploration;
* **exploration rate** is the parameter, controlling the distribution of the final policy. Setting it to a high value gives almost uniform distribution, while setting it to 0 makes us always select the best action.

The process of exploration is iterative. It starts with getting an **observation** from the game, and getting an **observation hash** using method *game.get_observation_str(observation)*.

```
    observation = game.get_observation(game_player)
    observation_str = game.get_observation_str(observation)
```

We iterate over **number of simulations**, each time performing exploration of our tree, until we find a previously unexplored or a teminal node using *search* method. Note that we should pass a **copy** of a current game state to the *search* method, as it changes the state of the game during exploration process.

```
    for i in range(self.num_simulations):
        game_clone = game.clone()
        _, value = self.search(game_clone, game_player)
```

So, the search begins. 
If we reached a terminal node, we just need to propagate a value for the current player:

```
    if game.is_ended():
        if game.is_draw():
            return True, -1
        return False, game.get_score(game_player)
```

If we reached a previously unexplored node, we get **P(s)** which is the prior probability of taking a particular action from state s according to the policy returned by our neural network. Note that the neural network agent uses the same interface **Agent** we defined earlier. 

```
    observation = game.get_observation(game_player)
    observation_str = game.get_observation_str(observation)
    valid_actions = game.get_valid_actions(game_player)

    # check if this observation is a previously unknown state
    if (observation_str not in self.Ps):
        # get an initial estimation of a policy and a value with a neural network-based agent
        self.Ps[observation_str], value = self.agent.predict(game, game_player)
        self.Ps[observation_str] = self.Ps[observation_str] * valid_actions  # masking invalid moves
        self.Ps[observation_str] /= np.sum(self.Ps[observation_str])  # renormalize
        self.Ns[observation_str] = 0
        return False, value
```

If we encountered a known state, we calculate **U(s, a)** values for every edge (action->state transition), which is an upper confidence bound on the Q value of our edge. These values are calculated as:

![alt text](http://i65.tinypic.com/ajw9jq.png)

```
    cur_best = -float('inf')
    best_act = -1

    # pick the action with the highest upper confidence bound
    for action in range(game.get_action_size()):
        if valid_actions[action]:
            if (observation_str, action) in self.Qsa:
                u = self.Qsa[(observation_str, action)] \
                        + self.cpuct * self.Ps[observation_str][action] * math.sqrt(self.Ns[observation_str]) / \
                            (1 + self.Nsa[(observation_str, action)])
            else:
                u = self.cpuct * self.Ps[observation_str][action] * math.sqrt(self.Ns[observation_str] + EPS)
        
            if u > cur_best:
                cur_best = u
                best_act = action
```

After reaching an unknown node and performing a neural network prediction, we propagate our values up the MCTS tree updating all the **Q(s, a)** values seen during the simulation. 

```
    # take the best action
    _, next_player = game.take_action(best_act)
    draw_result, value = self.search(game, next_player)

    # update values after search is done
    if game_player != next_player and not draw_result:
        value = -value

    if (observation_str, action) in self.Qsa:
        self.Qsa[(observation_str, action)] = (self.Nsa[(observation_str, action)] * self.Qsa[
                        (observation_str, action)] + value) / (self.Nsa[(observation_str, action)] + 1)
        self.Nsa[(observation_str, action)] += 1
    else:
        self.Qsa[(observation_str, action)] = value
        self.Nsa[(observation_str, action)] = 1
                    
    self.Ns[observation_str] += 1
    return False, value
```

Once we finished our MCTS simulations, **N(s, a)** values provide a good approximation for the policy from each state. The only thing left is to to apply our exploration rate parameter to control the distribution of the policy values:

```
    counts = [self.Nsa[(observation_str, a)] if (observation_str, a) in self.Nsa else 0 for a in
                  range(game.get_action_size())]

    if self.exp_rate == AgentMCTS.NO_EXPLORATION:
        bestA = np.argmax(counts)
        policy = [0] * len(counts)
        policy[bestA] = 1
    else:
        counts = [x ** (1. / self.exp_rate) for x in counts]
        policy = [x / float(sum(counts)) for x in counts]
    
    return policy, value
```

Congratulations! You understand now how the **Monte Carlo Tree Search** works! However, there is one more thing left, before we'll be able to make it work.

# Neural Network

Do you remember these lines from our AgentMCTS? 
```
    def __init__(self, agent, exp_rate=EXPLORATION_TEMP, cpuct=1, num_simulations=10):
        self.agent = agent
```

```
    self.Ps[observation_str], value = self.agent.predict(game, game_player)
```

That is exactly how we are going to use our neural network here. It produces policy and value, so the **Agent** interface may be reused with neural network too. Let's write a neural-network based Agent when! We'll use Keras library to make our code as clean as possible. 
First, we define our abstract neural network:

```
class NNet():
    def __init__(self, observation_size_x, observation_size_y, observation_size_z, action_size):
        self.observation_size_x = observation_size_x
        self.observation_size_y = observation_size_y
        self.observation_size_z = observation_size_z
        self.action_size = action_size
        
        self.model = self.build_model()
        
        self.graph = tf.get_default_graph()
    
    def build_model():
        '''
        Returns:
            model: a Keras model. Model gets observation with shape (self.observation_size_x, self.observation_size_y,  self.observation_size_z) and outputs a policy vector with size self.action_size and a value
        '''
        pass
        
    def predict(self, observation):
        with self.graph.as_default():
            self.model._make_predict_function()
            pi, v = self.model.predict(observation)

            if np.isscalar(v[0]):
                return pi[0], v[0]
            else:
                return pi[0], v[0][0]
```

And now we define our Agent which will use the neural network:

```
class AgentNNet(Agent):
    def __init__(self, nnet):
        self.nnet = nnet

    def predict(self, game, game_player):
        observation = game.get_observation(game_player)
        observation = observation[np.newaxis, :, :]

        return self.nnet.predict(observation)
```

In case you are wondering how to make neural network output a policy and a value, here is an example of a simple convolutional neural net for Alpha Zero: 

```
class ConvNNet(NNet):

    def build_model(self):
        num_channels = 512
        learning_rate = 0.0001

        input_boards = Input(shape=(self.observation_size_x, self.observation_size_y, self.observation_size_z))  # s: batch_size x board_x x board_y

        x_image = Reshape((self.observation_size_x, self.observation_size_y, self.observation_size_z))(input_boards)  # batch_size  x board_x x board_y x 1

        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(num_channels, 3, padding='same')(x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(
            Conv2D(num_channels, 3, padding='same')(h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='valid')(
            h_conv2)))  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(num_channels, 3, padding='valid')(
            h_conv3)))  # batch_size  x (board_x-4) x (board_y-4) x num_channels

        h_conv4_flat = Flatten()(h_conv4)

        s_fc1 = Dropout(0.3)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(0.3)(
            Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))  # batch_size x 1024

        pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  # batch_size x self.action_size
        v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        model = Model(inputs=input_boards, outputs=[pi, v])

        model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate))

        return model
```

# Bringing it all together 

Today we've defined our core interfaces: Game, Agent and NNet. We implemented a neural-network based Agent and a Monte Carlo Tree Search - based Agent! So, to initialize our Monte Carlo Agent we just need to write the code: 

```
    game = GameImplementation()
    observation_size = game.get_observation_size()

    nnet = ConvNNet(observation_size[0], observation_size[1], observation_size[2], game.get_action_size())
    agent_nnet = AgentNNet(nnet)

    agent_mcts = AgentMCTS(agent_nnet)
```

And our agent_mcts might predict the policy and value of a game state like this:

```

    policy, value = agent_mcts.predict(game, game.get_cur_player())

```

# What's next?

Awesome! We now have our core classes almost ready for writing an Alpha Zero training pipeline. Next time we are going to show you how to train an agent to play English draughts using Monte Carlo Tree Search and a deep residual network. The game has complexity of roughly **500,995,484,682,338,672,639** possible positions, so the task is going to be a fun one. 

Stay tuned!