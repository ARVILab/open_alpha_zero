# Building your own Alpha Zero. Part 1: Intuition

*"Fine, I'll do it myself" (c) Thanos, a supervillian from the MCU.*

In human society it’s considered to be natural to learn from the people which already have the knowledge you seek. In other words, to be the best, you must learn from the best. However, the recent Google gaming AI does not rely on any existing knowledge and prefers to discover every possible bit of knowledge in a game by itself.

As we all know, this strategy has lead to a massive success. Last year, Google published a paper describing the next generation of their gaming agents, the famous Alpha Zero, which was able to reach a super human level of play in go, chess and shogi. And it used only the knowledge generated via a lot of iterations of self-play!

## Introduction

Here at [ARVI Lab](http://ai.dev.arvilab.com/) we are building our own team of data scientists and mathematicians capable of applying recent breakthroughs in the ?eld   of AI to real-world problems. Unlike Alpha Zero itself, we are just mere humans, so we decided to learn from the best in our Reinforcement Learning research. In order to learn a state-of-the-art in the ?eld, we implemented Alpha Zero using our own resources and we are going to share our code and experience in a series of articles.

So, without further boring introductions, let’s begin our journey with the Part One: Intuition!

## Learning to play a game without any existing knowledge

*"Live. Die. Repeat."*

Imagine yourself learning a competitive game, when no one have ever bothered to tell you the rules. What is even more sad, no one wants to play with you, so you don’t even have a partner to train with. All you have ever been tough is the set of valid actions you might take and you also get a noti?cation the moment you lose or win a game session. Also, you can get an observation of the game in form of an image on every timestep of the game.

Your task is simple. You need to become the best player in the history of the game.

However, you have one useful trick up your sleeve. You can clone yourself, producing an exact same copy.

So, you begin training. First of all, you clone yourself because you need a sparring partner. When you play with the clone, simply picking a random valid action every time you have to. Every time you take an action, you record a current state of the game and the action you took. At some point, you ?nally lose or win.

When you look into your recordings and try to ?gure out what actions were the right ones. You also try to understand why the last state of the game was terminal and how to avoid it, simultaneously trying to push your opponent into his terminal state. Once you feel like you are ready to give it another shot, you clone yourself one more time and play again.

Every time you learn from a new game session, you become a little bit better. At one point you stop just picking random actions and start using strategies. However, you still take random actions sometimes, to see if there are any undiscovered opportunities for you to exploit, so you eventually learn how to overcome the strategies you’ve developed on previous iterations.

You repeat the process a few million times. When you are done, no one can defeat you in the game.

Congratulations! You understand now how the Alpha Zero learns!

## The core components

There is a great Alpha Zero cheatsheet published in [Applied Data Science blog](https://medium.com/applied-data-science).

![Alpha Zero](https://cdn-images-1.medium.com/max/1000/1*0pn33bETjYOimWjlqDLLNw.png)

Does it still look a bit overwhelming? Don't worry, we got you covered! Let's break it down in a simple manner.

At the core of an AI **agent**, based on the Alpha Zero, there are two key components:
* A **neural network**, capable of generating a **stochastic policy** and a **continuous value** of a game state state given an **observation** of the state;
* A **Monte Carlo Tree Search** implementation which helps to improve the **policy** of the **agent** via simulations of future states of the game

An **environment** (game) which the **agent** is trying to solve should provide:
* A vector of **possible actions** where every action is defined by it's index;
* A vector of **valid actions** on every timestep. The list has similar length as the vector of all actions and every possible action is encoded as **1** if it's available or **0** otherwise;
* An **observation** of the game state from the point of view of the player which should take his action next. The observation might be encoded as a vector or a matrix;
* A notification if the **terminal state** of the game is reached and an index of the player who won the game;

To put it simple, the stochastic **policy** is a vector of a similar length as a vector of possible actions, but every element of the vector represnts a probability of an action. The bigger the probability, the better the action from the neural network's point of view. The continuous **value** of a game state is a number between [-1, 1] which represents likehood of the given player winning the game from the current state.

The **Monte-Carlo Tree Search** (MCTS) algorithm serves as an "imagination" of the **agent**. MCTS tries to predict what is going to happen next by using the neural network to predict next moves of the opponent. 

## Piecing it all together

First, we initialize our neural network with random values. The **inputs** of the neural network are of the same shape as an **observation** of the game, and the **outputs** are a policy vector and a value of the state, given the observation.

When we start our first iterarion of the self-play. On every game step we get an **observation** from the game **environment** and feed it to the **Monte-Carlo Tree Search** (MCTS). The MCTS algorithm uses the **neural network** to simulate next **number of MCTS simulations** steps of the game and when returns us an updated **policy**. We pick an action from the policy (remember: it's a distribution of actions probabilities) and move to the next game step. On every game step we collect a pair: **policy** and **obeservation**. 
When the game is finished, we update our collected pairs by appending a game result to them. All of the pairs of the winner get 1, and all of the pairs of the loser get -1.

When we feed a list of trios (**policy**, **obvervation**, **result**) to the neural network and train it. The training set is built from observations and the labels set is built from policies and results.

By repeating the process we increase the quality of our neural networks estimations which leads to better MCTS estimations. The better MCTS estimations we get, the better strategies our agent learns.

## What's next? 

The idea behind Alpha Zero is pretty simple. However, the process of it's implementation presents a lot of challanges, tricks and bottlenecks which a developer should overcome in order to train Alpha Zero effeciently. 
In our next article we will dive deeper into neural network design and monte carlo tree search in order to uncover and solve these challanges. 
Stay tuned!