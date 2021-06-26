#!/usr/bin/env bash

cd /works/resources
git clone -b v1.28.1 https://github.com/grpc/grpc
cd grpc && git submodule update --init
mkdir -p cmake/build && cd cmake/build
cmake ../.. && make && make install
cd ../../examples/cpp/helloworld
mkdir -p cmake/build