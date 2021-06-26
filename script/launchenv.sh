#!/usr/bin/env bash

sudo docker run -itd \
                --cap-add sys_ptrace \
                -p 2222:22 \
                -p 3333:33 \
                -p 4444:44 \
                -p 5555:55 \
                -p 6666:66 \
                -v `pwd -P`/works:/works \
                --name denv \
                nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 \
                /bin/bash

apt-get update
apt-get install git
apt-get install wget
wget -q -O cmake-linux.sh https://github.com/Kitware/CMake/releases/download/v3.17.0/cmake-3.17.0-Linux-x86_64.sh
sh cmake-linux.sh -- --skip-license --prefix=/usr
