#!/bin/bash
COLOR_RED='\033[0;31m'
if [ $# == 1 ]; then
  if [ $1 == "run" ] ; then
    echo "docker run --privileged --net=host --env=DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --volume=$HOME/.Xauthority:/root/.Xauthority:rw -it -v $HOME/program/DL_Pytorch:/workspace --name dl_container iles88039/pytorch:v2.2"
    # --env=DISPLAY:
    # share the Host’s DISPLAY environment variable to the Container
    # -v /tmp/.X11-unix:/tmp/.X11-unix:
    # mount the X11 socket
    # --volume=$HOME/.Xauthority:/root/.Xauthority:rw:
    # share the Host’s XServer with the Container
    docker run --privileged --net=host --env="DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -it -v $HOME/program/DL_Pytorch:/workspace --name dl_container iles88039/pytorch:v2.2
  elif [ $1 == "start" ] ; then
    echo "docker start dl_container"
    docker start dl_container
  elif [ $1 == "exec" ] ; then
    echo "docker exec -it dl_container bash"
    docker exec -it dl_container bash
  else
    echo -e "${COLOR_RED}Usage: source cmd.sh [run | start | exec]${COLOR_NC}"
  fi
else
    echo -e "${COLOR_RED}Usage: source cmd.sh [run | start | exec]${COLOR_NC}"
fi
