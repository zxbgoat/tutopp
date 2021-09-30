add_executable(wrivid writevideo.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(wrivid opencv_core opencv_videoio)