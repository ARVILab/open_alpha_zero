# Alpha Zero by [ARVI Lab](http://ai.arvilab.com/).

Train an AI to play any game with Python interface using either Alpha Go or Alpha Go Zero training process. 
The main goal of the project is to maximize training efficiency using regular machines and GPUs. The features of the project:
* Build your own neural network in Keras
* Optimize your neural network for inference using one line of code
* Distribute your self-play generation across multiple machines and GPUs via configurations
* Distribute your neural network training across multiple machines and GPUs using Keras distributed training or Uber's Horovod
* Train your AI to play games with both complete (go, chess, checkers) and incomplete informations (card games)

## Getting started

The project has built-in support of two games:
* a board game [american checkers](https://simple.wikipedia.org/wiki/Checkers), also known as draughts;
* a card game [durak](https://en.wikipedia.org/wiki/Durak);

We strongly recommend to read the series of articles about our project:
* [Builing your own Alpha Zero: Introduction](http://ai.arvilab.com/post/building-your-own-alpha-zero-part-1-intuition)
* [Builing your own Alpha Zero: Decision Making](http://ai.arvilab.com/post/building-your-own-alpha-zero-part-2-decisions)
* Builing your own Alpha Zero: Training
* Builing your own Alpha Zero: Distributed training and optimization

### Project's core interfaces

Three main entities in the project are the [Agent](core/interfaces/Agent.py), the [Game](core/interfaces/Game.py) and the [EnvironmentSelector](EnvironmentSelector.py). 

The **Agent** interface covers any entity which is capable of predicting actions of a player in a game. The Agents differ on their approach of decision making. The project includes:
* A [random Agent](core/agents/AgentRandom.py)
* A [neural network-based Agent](core/agents/AgentNNet.py)
* A [MCTS-based Agent](core/agents/AgentMCTS.py)

It also has checkers-specific [Alpha Beta pruning-based Agent](games/checkers/agent/AgentAlphaBeta.py) and an [Agent](games/checkers/agent/CheckersHumanAgent.py) which serves as a proxy to the human input from a terminal.

The **Game** interface defines the environment in which an Agent operates. To create your own environment implement a [Game](core/interfaces/Game.py) interface to your game. The examples of the Game are:
* [Checkers Game](games/checkers/CheckersGame.py)
* [Durak Game](games/durak/DurakGame.py)

The **EnvironmentSelector** class let's user store different configurations of games and agents. These configurations are named "profiles". To add your game and your agent to the project you must add a profile to EnvironmentSelector and a build function for your configuration.

The main scripts of the project are [alpha zero trainer](alpha_zero_trainer.py) and [pit](pit.py)

#### Alpha zero trainer

The script has two **required** options:
* **"--agent"** specifies agent profile (Game + trainable Agent)
* **"--workspace"** specifies the folder where all of the training-specific files will be stored

It also has options to control the number of games played on self-play stage and evaluation stages:
* **"--iterations"** specifies how many iterations of self-play will be played
* **"--games_num"** specifies how many games will be played on each iteration
* **"--test_games_num"** specifies how many evaluation games will be played on each stage of the game
* **"--skip_evaluation"** switches the project into Alpha Zero approach of training. The default approach to training is based on Alpha Go approach. 

To continue training from the checkpoint you should specify two options:
* **"--agent_path"** specifies path ot agent's checkpoint. You will usually use the path "workspace/best.h5"
* **"--memory_path"** specifies path to the games your agent generated. You will usually use the path "workspace/memory.pkl"

A user may also control a level of verbosity of the script by using:
* **"--verbose"** option let's user see the outcome of each game played
* **"--debug"** option let's user see every turn of an every game played

You may also limit number of steps in the game by specifying **"--max_steps"** option which is useful in case of an possibly "infinite" game like checkers. 
To limit exploration in self-play stage, use **"--exploration_decay_steps"** option. After specified number of the steps, an agent will choose only best actions which may help to learn different openings in a game but does not interferes with a late stages of the game.

To speed up your training process you may use option **"--optimize_for_inference"** which will optimize you model on each iteration reducing the time of decision making. 

If you want to speed up training even further you may use additional machines and gpus. Our tutorial about [distributed training](wiki/distributed_training.md) got you covered!

#### Pit script

The script is used to check how Agents are performing agains each other. It uses a lot of similar options, like  --games_num, --verbose, --debug, --max_steps, --optimize_for_inference. To specify the opponents use options: --agent_new, --agent_new_path, --agent_old_path, --agent_old_path. Note: you don't have to use "path" options in case of non-neural network based agents!

### Train Alpha Zero to play checkers

The following configuration will train checkes synchroniously on one gpu. The agent will generate 10 000 games using profile "checkers_agent_train_rcnn_default" from EnvironmentSelector.

```
python3 alpha_zero_trainer.py \
--agent "checkers_agent_train_rcnn_default" \
--workspace "games/checkers/training" \
--max_steps 400 \
--games_num 100 \
--iterations 100 \
--verbose \
--exploration_decay_steps 15
```

After the training has been finished, you may find a lot of models in "games/checkers/training" folder. You may verify how the Agent performs using "pit" script as follows:
```
python3 pit.py \
--agent_new "checkers_agent_train_rcnn_default" \
--agent_old "checkers_agent_random" \
--agent_new_path "games/checkers/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100
```

The script will play 100 games using your trained model against a random agent. A trained agent will win in every game agains such a weak opponent. You might as well use these scripts to try your agent against Alpha Beta pruning algorithm and even against yourself:
```
python3 pit.py \
--agent_new "checkers_agent_train_rcnn_default" \
--agent_old "checkers_agent_alpha_beta" \
--agent_new_path "games/checkers/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

python3 pit.py \
--agent_new "checkers_agent_train_rcnn_default" \
--agent_old "checkers_agent_human" \
--agent_new_path "games/checkers/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

```

You may speed up the decision making of your agent by adding "--optimize_for_inference" option to the pit command.

### Train Alpha Zero to play durak

The following configuration will train durak synchroniously on one gpu. The agent will generate 10 000 games using profile "durak_agent_train_default" from EnvironmentSelector.

```
python3 alpha_zero_trainer.py \
--agent "durak_agent_train_default" \
--workspace "games/durak/training" \
--max_steps 400 \
--games_num 100 \
--iterations 100 \
--verbose \
--exploration_decay_steps 15
```

After the training has been finished, you may find a lot of models in "games/durak/training" folder. You may verify how the Agent performs using "pit" script as follows:
```
python3 pit.py \
--agent_new "durak_agent_train_default" \
--agent_old "durak_agent_random" \
--agent_new_path "games/durak/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100
```

The script will play 100 games using your trained model against a random agent. A trained agent will win in every game agains such a weak opponent. You might as well use these scripts to try your agent against random agent and even against yourself:
```
python3 pit.py \
--agent_new "durak_agent_train_default" \
--agent_old "durak_agent_random" \
--agent_new_path "games/durak/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

python3 pit.py \
--agent_new "durak_agent_train_default" \
--agent_old "durak_agent_human" \
--agent_new_path "games/durak/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

```

You may speed up the decision making of your agent by adding "--optimize_for_inference" option to the pit command.

### Train Alpha Zero to play tictactoe

The following configuration will train tictactoe synchroniously on one gpu. The agent will generate 10 000 games using profile "tictactoe_agent_train_default" from EnvironmentSelector.

```
python3 alpha_zero_trainer.py \
--agent "tictactoe_agent_train_default" \
--workspace "games/tictactoe/training" \
--max_steps 400 \
--games_num 100 \
--iterations 100 \
--verbose \
--exploration_decay_steps 15
```

After the training has been finished, you may find a lot of models in "games/tictactoe/training" folder. You may verify how the Agent performs using "pit" script as follows:
```
python3 pit.py \
--agent_new "tictactoe_agent_train_default" \
--agent_old "tictactoe_agent_random" \
--agent_new_path "games/tictactoe/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100
```

The script will play 100 games using your trained model against a random agent. A trained agent will win in every game agains such a weak opponent. You might as well use these scripts to try your agent against random agent and even against yourself:
```
python3 pit.py \
--agent_new "tictactoe_agent_train_default" \
--agent_old "tictactoe_agent_random" \
--agent_new_path "games/tictactoe/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

python3 pit.py \
--agent_new "tictactoe_agent_train_default" \
--agent_old "tictactoe_agent_human" \
--agent_new_path "games/tictactoe/training/best.h5" \
--verbose \
--debug \
--max_steps 400 \
--games_num 100

```

You may speed up the decision making of your agent by adding "--optimize_for_inference" option to the pit command.

## Basic installation

Note: the project was tested only on Ubuntu 16.04. Compatility with other OS is not guaranteed. 

### 1. Install Python 3.5+

### 2. Install CUDA 9.0

Follow the guide: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### 3. Install CUDNN 7.0.5

Follow the guide: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/

### 4. Install Tensorflow-gpu

Follow the guide: https://www.tensorflow.org/install/install_linux

### 5. Install requirements

Type the command:

```

pip3 install -r requirements.txt

```

### 6. Install OpenMPI

Type the commands outside of project's folder:

```

wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.1.tar.gz
gunzip -c openmpi-3.0.1.tar.gz | tar xf -
cd openmpi-3.0.1
./configure --prefix=/usr/local
make
sudo make install

export PATH="$PATH:/usr/local/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/"

echo export PATH="$PATH:/usr/local/bin" >> /home/ubuntu/.bashrc
echo export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib/" >> /home/ubuntu/.bashrc

```

### 7. Install NCCL2

Download NCCL2 2.0.5 from https://developer.nvidia.com/nccl

Type the commands: 

```

sudo dpkg -i nccl-repo-<version>.deb
sudo apt update
sudo apt install libnccl2 libnccl-dev

```

### 8. Install Uber's Horovod

```

HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir horovod --user

```

## Cluster setup

This project supports distributed training and self-play generation. The project might run in following distributed modes: 
* Master - client. Self-play generation and neural network training are distributed between all of the machines (and their GPU's) in a cluster. NOTE: this feature requires at least 25 Gb/s network bandwidth between the machines!
* Master - client. Self-play generation is distributed between all of the machines (and their GPU's) in a cluster. Neural network training is performed on master machine in distributed mode (between GPU on the master machine). 

### 1. Configure SSH communication

Make your remote machines available to direct ssh communication without keys. Check it by typing commands like the one below to every client machine in your cluster from your master machine: 

```

ssh <ip_to_worker_machine>

```  

### 2. Setting up NFS on a master machine

```

sudo apt-get install nfs-kernel-server
mkdir cloud
cat /etc/exports /home/mpiuser/cloud *(rw,sync,no_root_squash,no_subtree_check)
exportfs -a
sudo service nfs-kernel-server restart

```

### 3. Setting up NFS on a client machine

```

sudo apt-get install nfs-common
mkdir cloud
sudo mount -t nfs master:/home/user/cloud ~/cloud

```

To make the mount permanent so you don’t have to manually mount the shared directory everytime you do a system reboot, you can create an entry in your file systems table - i.e., /etc/fstab file like this:

```

cat /etc/fstab
#MPI CLUSTER SETUP
master:/home/mpiuser/cloud /home/mpiuser/cloud nfs

```

## Contributors and Credits

* This project was initially based on the repo [Alpha Zero General](https://github.com/suragnair/alpha-zero-general).
* This project utilizes [OpenMPI](https://www.open-mpi.org).
* This project utilizes [Uber's Horovod](https://github.com/uber/horovod).
