#add_library(fused_attention STATIC test.cpp test.h)
add_library(test STATIC test.hpp)
target_link_libraries(test PRIVATE composable_kernel::device_operations)
