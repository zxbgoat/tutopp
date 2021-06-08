# tensorrt
include_directories(/usr/local/tensorrt/include)
include_directories(/usr/local/tensorrt/samples/common)
link_directories(/usr/local/tensorrt/lib)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/thirdpart)

add_library(logger SHARED /usr/local/tensorrt/samples/common/logger.cpp)
# add_library(scratch SHARED mnist_scratch.cpp)
# add_library(onnx SHARED mnist_onnx.cpp)
add_library(plugin SHARED mnist_plugin.cpp)
add_library(dynashape SHARED mnist_dynashape.cpp)
add_executable(mnist mnist.cpp)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(dynashape logger nvonnxparser cudart)
target_link_libraries(mnist dynashape)
