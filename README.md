# Exercise_Pytorch
Implement some deep learning exercises with pytorch.

## How to Setup Environment
### a. Use docker to setup Pytorch environment directly.
```
$ docker pull iles88039/pytorch:v2
```
Please refer to the Dockerfile if you are interested.\
You can also visit my dockerhub for more details:
https://hub.docker.com/repository/docker/iles88039/pytorch \
Pytorch dockerhub:
https://hub.docker.com/r/pytorch/pytorch \
### b. Clone this repository in $HOME/program/DL_Pytorch directory.
```
$ mkdir -p ~/program/DL_Pytorch
$ cd ~/program/DL_Pytorch
$ mkdir data
$ git clone https://github.com/BrianCmHunag/Exercise_Pytorch.git
```
### c. Type the cmd to enter docker container.
```
$ cd ~/program/DL_Pytorch/Exercise_Pytorch
$ source cmd.sh [run | start | exec]
```
run: Setup a new container named "dl_container".\
start: If the container was stopped, it will turn on.\
exec: Enter the cmd shell.\
