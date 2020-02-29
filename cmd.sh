#!/bin/bash
COLOR_RED='\033[0;31m'
if [ $# == 1 ]; then
  if [ $1 == "run" ] ; then
    echo "docker run --privileged -it -v $HOME/program/Exercise_Pytorch/workspace:/workspace --name dl_container pytorch/pytorch"
    docker run --privileged -it -v $HOME/program/Exercise_Pytorch/workspace:/workspace --name dl_container pytorch/pytorch
  elif [ $1 == "start" ] ; then
    echo "docker start -it dl_container bash"
    docker start dl_container
  elif [ $1 == "exec" ] ; then
    echo "docker start -it dl_container bash"
    docker exec -it dl_container bash
  else
    echo -e "${COLOR_RED}Usage: source cmd.sh [run | start | exec]${COLOR_NC}"
  fi
else
    echo -e "${COLOR_RED}Usage: source cmd.sh [run | start | exec]${COLOR_NC}"
fi
