// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using InDataType  = int8_t;
using WeiDataType = int8_t;
using AccDataType = int32_t;
using OutDataType = int8_t;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using InElementOp  = ck::tensor_operation::element_wise::PassThrough;
using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
using OutElementOp = ck::tensor_operation::element_wise::PassThrough;

using InLayout  = ck::tensor_layout::convolution::GNHWC;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using OutLayout = ck::tensor_layout::convolution::GNHWK;

static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto Filter1x1Pad0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
static constexpr auto Filter1x1Stride1Pad0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

static constexpr auto GemmPadingSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

using device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_instances = std::tuple<
    // clang-format off
           // ###############################|        NDim|     InData|     WeiData|     OutData|     AccData| InLayout| WeiLayout| OutLayout|           In|           Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
           // ###############################|     Spatial|       Type|        Type|        Type|        Type|         |          |          |  Elementwise|   Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
           // ###############################|            |           |            |            |            |         |          |          |    Operation|     Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
           // ###############################|            |           |            |            |            |         |          |          |             |              |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGroupedConvFwdDl_NHWC_KYXC_NHWK<           2, InDataType, WeiDataType, OutDataType, AccDataType, InLayout, WeiLayout, OutLayout,  InElementOp,  WeiElementOp,  OutElementOp,       ConvSpec,    GemmPadingSpec,   256,   128,   128,    16,  4,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

using device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_Filter1x1Pad0_instances = std::tuple<
    // clang-format off
           // ###############################|        NDim|     InData|     WeiData|     OutData|     AccData| InLayout| WeiLayout| OutLayout|           In|           Wei|           Out|    Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
           // ###############################|     Spatial|       Type|        Type|        Type|        Type|         |          |          |  Elementwise|   Elementwise|   Elementwise|        Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
           // ###############################|            |           |            |            |            |         |          |          |    Operation|     Operation|     Operation| Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
           // ###############################|            |           |            |            |            |         |          |          |             |              |              |               |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGroupedConvFwdDl_NHWC_KYXC_NHWK<           2, InDataType, WeiDataType, OutDataType, AccDataType, InLayout, WeiLayout, OutLayout,  InElementOp,  WeiElementOp,  OutElementOp,  Filter1x1Pad0,    GemmPadingSpec,   256,   128,   128,    16,  4,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
    // clang-format on
    >;

using device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_Filter1x1Stride1Pad0_instances =
    std::tuple<
        // clang-format off
           // ###############################|        NDim|     InData|     WeiData|     OutData|     AccData| InLayout| WeiLayout| OutLayout|           In|           Wei|           Out|          Convolution|              GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|      ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|      BBlockTransfer|     CThreadTransfer| CThreadTransfer|    CThreadTransfer|
           // ###############################|     Spatial|       Type|        Type|        Type|        Type|         |          |          |  Elementwise|   Elementwise|   Elementwise|              Forward|    Spacialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|     DstVectorTensor|        SrcDstAccess| SrcDstVectorDim| DstScalarPerVector|
           // ###############################|            |           |            |            |            |         |          |          |    Operation|     Operation|     Operation|       Specialization|                  |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder| Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder| Lengths_K0_N0_N1_K1|               Order|                |                   |
           // ###############################|            |           |            |            |            |         |          |          |             |              |              |                     |                  |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                    |                   |                     |               |               |                    |                   |                    |                    |                |                   |
        DeviceGroupedConvFwdDl_NHWC_KYXC_NHWK<           2, InDataType, WeiDataType, OutDataType, AccDataType, InLayout, WeiLayout, OutLayout,  InElementOp,  WeiElementOp,  OutElementOp, Filter1x1Stride1Pad0,    GemmPadingSpec,   256,   128,   128,    16,  4,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>,      S<8, 1, 1, 4>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 4>,      S<1, 2, 0, 3>,       S<1, 1, 1, 4>, S<0, 1, 2, 3, 4, 5>,               5,                 4>
        // clang-format on
        >;

void add_device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwd<2,
                                                     InLayout,
                                                     WeiLayout,
                                                     OutLayout,
                                                     InDataType,
                                                     WeiDataType,
                                                     OutDataType,
                                                     InElementOp,
                                                     WeiElementOp,
                                                     OutElementOp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_instances{});

    add_device_operation_instances(
        instances, device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_Filter1x1Pad0_instances{});

    add_device_operation_instances(
        instances,
        device_grouped_conv2d_fwd_dl_gnhwc_gkyxc_gnhwk_int8_Filter1x1Stride1Pad0_instances{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
