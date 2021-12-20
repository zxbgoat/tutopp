cmake_minimum_required(VERSION 3.17)
# project(tuto CUDA CXX)
project(tuto CXX)

# tensorrt
include_directories(/usr/local/tensorrt/include
                    /usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# add_subdirectory(src/tensorrt/mnist)
# add_subdirectory(src/utils/test)
# add_subdirectory(src/grpc/fetchimg)
# add_subdirectory(src/opencv/writevid)
# add_subdirectory(src/boost/datetime.cpp)
# add_subdirectory(src/cuda)
# add_subdirectory(source/tensorrt/mnistint8)
# add_subdirectory(source/utils/registry)
# add_subdirectory(source/torch)
add_subdirectory(source/design-pattern)
