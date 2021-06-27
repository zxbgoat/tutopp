cmake_minimum_required(VERSION 3.17)
project(tuto CUDA CXX)

add_subdirectory(src/tensorrt/mnist)
add_subdirectory(src/utils/test)
add_subdirectory(src/grpc/fetchimg)
