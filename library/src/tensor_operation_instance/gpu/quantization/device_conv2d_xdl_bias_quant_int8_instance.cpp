// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using GNHWC       = ck::tensor_layout::convolution::GNHWC;
using GKYXC       = ck::tensor_layout::convolution::GKYXC;
using GNHWK       = ck::tensor_layout::convolution::GNHWK;
using GK          = ck::tensor_layout::convolution::G_K;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using Relu        = ck::tensor_operation::element_wise::Relu;

using GK_Tuple  = ck::Tuple<GK>;
using I32_Tuple = ck::Tuple<int32_t>;

using Add_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<PassThrough>;
using Add_Relu_Mul_Clamp = ck::tensor_operation::element_wise::Add_Activation_Mul_Clamp<Relu>;

static constexpr ck::index_t NDimSpatial = 2;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
static constexpr auto ConvFwdDefault =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto ConvFwd1x1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
static constexpr auto ConvFwd1x1S1P0 =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;

// TODO - Add more instances
template <typename OutElementOp, ConvolutionForwardSpecialization ConvSpec>
// clang-format off
using device_conv2d_int8_instances =
    std::tuple <
        //########################################|  NumDim|      A|      B|       Ds|      E|  AData|  BData| AccData| CShuffle|        Ds|  EData|           A|           B|          CDE|    ConvForward|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //########################################| Spatial| Layout| Layout|   Layout| Layout|   Type|   Type|    Type| DataType|  DataType|   Type| Elementwise| Elementwise|  Elementwise| Specialization| Specialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //########################################|        |       |       |         |       |       |       |        |         |          |       |   Operation|   Operation|    Operation|               |               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //########################################|        |       |       |         |       |       |       |        |         |          |       |            |            |             |               |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   256,   256,   128,    64,  16,  16,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   256,   128,   256,    64,  16,  16,   32,   32,    2,    4,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   128,   128,   128,    64,  16,  16,   32,   32,    4,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   256,   128,   128,    64,  16,  16,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   128,   128,    64,    64,  16,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   128,    64,   128,    64,  16,  16,   32,   32,    2,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,    64,    64,    64,    64,  16,  16,   32,   32,    2,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   256,   128,    64,    64,  16,  16,   32,   32,    2,    1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   256,    64,   128,    64,  16,  16,   32,   32,    1,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   128,   128,    32,    64,  16,  16,   32,   32,    2,    1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 64, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,   128,    32,   128,    64,  16,  16,   32,   32,    1,    2,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 32, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 4>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,    64,    64,    32,    64,  16,  16,   32,   32,    2,    1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 2>,               8>,
        DeviceGroupedConvFwdMultipleD_Xdl_CShuffle<       2,  GNHWC,  GKYXC, GK_Tuple,  GNHWK, int8_t, int8_t, int32_t,  int32_t, I32_Tuple, int8_t, PassThrough, PassThrough, OutElementOp,       ConvSpec,       GemmSpec,        1,    64,    32,    64,    64,  16,  16,   32,   32,    1,    2,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,             16,             16,         1,     S<4, 16, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,             16,             16,         1,           1,           1,               S<1, 32, 1, 2>,               8>
    >;
// clang-format on

void add_device_conv2d_bias_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              ck::Tuple<GK>,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              ck::Tuple<int32_t>,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Add_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(instances,
                                   device_conv2d_int8_instances<Add_Mul_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(instances,
                                   device_conv2d_int8_instances<Add_Mul_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(instances,
                                   device_conv2d_int8_instances<Add_Mul_Clamp, ConvFwd1x1S1P0>{});
}

void add_device_conv2d_bias_relu_perlayer_quantization_int8_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleD<NDimSpatial,
                                                              GNHWC,
                                                              GKYXC,
                                                              ck::Tuple<GK>,
                                                              GNHWK,
                                                              int8_t,
                                                              int8_t,
                                                              ck::Tuple<int32_t>,
                                                              int8_t,
                                                              PassThrough,
                                                              PassThrough,
                                                              Add_Relu_Mul_Clamp>>>& instances)
{
    add_device_operation_instances(
        instances, device_conv2d_int8_instances<Add_Relu_Mul_Clamp, ConvFwdDefault>{});
    add_device_operation_instances(
        instances, device_conv2d_int8_instances<Add_Relu_Mul_Clamp, ConvFwd1x1P0>{});
    add_device_operation_instances(
        instances, device_conv2d_int8_instances<Add_Relu_Mul_Clamp, ConvFwd1x1S1P0>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
