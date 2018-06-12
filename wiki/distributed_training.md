# Distributed training

## Cluster setup

This project supports distributed training and self-play generation. The project might run in following distributed modes:
1) One machine with multiple GPUs. Self-play generation and Neural network training are distributed between all of the GPUs on master machine.
2) Multiple machines, multiple GPUs (master -> client). Self-play generation is distributed between all of the machines (and their GPU's) in a cluster. Neural network training is performed on master machine in distributed mode (between GPUs on the master machine).
3) Multiple machines, multiple GPUs, good network speed (master <-> client). Self-play generation and neural network training are distributed between all of the machines (and their GPU's) in a cluster. 
**Warning**: this feature requires at least 25 Gb/s network bandwidth between the machines!


### 1. Configure SSH communication

Make your remote machines available to direct ssh communication without keys. Check it by typing commands like the one below to every client machine in your cluster from your master machine: 

```

ssh <ip_to_client_machine>

```  

If the command fails, use [this tutorial](http://www.linuxproblem.org/art_9.html) to make direct communication between master and client machines via ssh. 

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

To make the mount permanent so you don't have to manually mount the shared directory every time you do a system reboot, you can create an entry in your file systems table - i.e., /etc/fstab file like this:

```

cat /etc/fstab
#MPI CLUSTER SETUP
master:/home/mpiuser/cloud /home/mpiuser/cloud nfs

```

## Configuring distributed training process

### One machine with multiple GPUs
Add following options to your **alpha_zero_trainer.py** calling parameters:

```

--hosts "your_master_IP_addr:number_of_gpus_on_master" \
--train_distributed_native

```

This will result in self-play generation distributed over all of the GPUs on master machine. So, if you will call **alpha_zero_trainer.py** with the parameter *"--games_num 100"* on a machine with 2 GPUs, every GPU will be used to generate 50 games. Neural network training will be distributed using Keras API between *"number_of_gpus_on_master"* GPUs.

### Multiple machines, multiple GPUs (master -> client)
Add following options to your **alpha_zero_trainer.py** calling parameters:

```

--hosts "your_master_IP_addr:number_of_gpus_on_master,your_client_IP_addr:number_of_gpus_on_client" \
--train_distributed_native

```

This will result in self-play generation distributed over all of the GPUs on master and client machines. So, if you will call **alpha_zero_trainer.py** with the parameter *"--games_num 100"* on a master machine with 2 GPUs and a client machine with 2 gpus, every GPU will be used to generate 25 games. Neural network training will be distributed using Keras API between *"number_of_gpus_on_master"* GPUs.

### Multiple machines, multiple GPUs, good network speed (master <-> client)
Add following options to your **alpha_zero_trainer.py** calling parameters:

```

--hosts "your_master_IP_addr:number_of_gpus_on_master,your_client_IP_addr:number_of_gpus_on_client" \
--train_distributed

```

This will result in self-play generation distributed over all of the GPUs on master and client machines. So, if you will call **alpha_zero_trainer.py** with the parameter *"--games_num 100"* on a master machine with 2 GPUs and a client machine with 2 gpus, every GPU will be used to generate 25 games. Neural network training will be distributed using [Horovod API](https://github.com/uber/horovod) between *"number_of_gpus_on_master + number_of_gpus_on_client"* GPUs.