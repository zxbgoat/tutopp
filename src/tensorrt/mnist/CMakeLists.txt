# tensorrt
include_directories(/usr/local/tensorrt/include)
include_directories(/usr/local/tensorrt/samples/common)
link_directories(/sur/local/tensorrt/lib)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/thirdpart)

add_library(scratch SHARED mnist_scratch.cpp)
add_library(plugin SHARED mnist_plugin.cpp)
add_library(onnx SHARED mnist_onnx.cpp)
add_library(dynashape SHARED mnist_dynashape.cpp)
add_executable(mnist mnist.cpp)
