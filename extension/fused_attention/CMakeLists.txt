#add_library(fused_attention STATIC fused_attention.cpp fused_attention.h)
add_library(fused_attention STATIC fused_attention.hpp)
target_link_libraries(fused_attention PRIVATE composable_kernel::device_operations)
