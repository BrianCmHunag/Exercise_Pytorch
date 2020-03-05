#!/bin/bash
COLOR_RED='\033[0;31m'
if [ $# == 1 ]; then
  if [ $1 == "run" ] ; then
    echo "docker run --privileged --env=DISPLAY -it -v $HOME/program/DL_Pytorch:/workspace --name dl_container pytorch/pytorch"
    docker run --privileged --env="DISPLAY" -it -v $HOME/program/DL_Pytorch:/workspace --name dl_container iles88039/pytorch:v2
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
