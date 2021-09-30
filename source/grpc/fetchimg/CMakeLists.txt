# opencv grpc
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# link_directories(/usr/local/acl/lib_acl/lib/
#                  /usr/local/acl/lib_acl_cpp/lib/)

add_executable(fetch fetchimg.cpp
                     image.pb.cc
                     image.grpc.pb.cc)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries(fetch grpc++ grpc grpc++ protobuf opencv_core opencv_imgproc opencv_imgcodecs z upb ssl cares gpr
        crypto pthread absl_bad_optional_access absl_base absl_dynamic_annotations absl_int128
        absl_log_severity absl_raw_logging_internal absl_spinlock_wait absl_str_format_internal
        absl_strings absl_strings_internal absl_throw_delegate address_sorting dl)