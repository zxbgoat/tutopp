# tensorrt
include_directories(/usr/local/tensorrt/include)
include_directories(/usr/local/tensorrt/samples/common)
link_directories(/sur/local/tensorrt/lib)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_library(scratch SHARED mnist_scratch.cpp)
add_library(plugin SHARED mnist_plugin.cpp)
add_executable(mnist mnist.cpp)
