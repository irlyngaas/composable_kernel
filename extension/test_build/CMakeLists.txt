#add_library(fused_attention STATIC test_build.cpp test_build.h)
add_library(test_build STATIC test_build.hpp)
target_link_libraries(test_build PRIVATE composable_kernel::device_operations)
