// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <array>

#include "ck/tensor_operation/gpu/device/device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Convolution Forward:
//   input : input image A[G, N, C, Hi, Wi],
//   input : weight B[G, K, C, Y, X],
//   input : D0[G, N, K, Ho, Wo], D1[G, N, K, Ho, Wo], ...
//   output : output image E[G, N, K, Ho, Wo]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct DeviceGroupedConvFwd : public BaseOperator
{
    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a, // input image
                        const void* p_b, // weight
                        void* p_c,       // output image
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CElementwiseOperation& c_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
