#include "python/tools/kernel_explorer/device_array.h"

#include <iostream>
#include <vector>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using CElementOp    = ck::tensor_operation::element_wise::PassThrough;

constexpr static auto MaskingSpec =
    ck::tensor_operation::device::MaskingSpecialization::MaskDisabled;

using ADataType   = ck::half_t;
using B0DataType  = ck::half_t;
using B1DataType  = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

void fused_attention( const int G0, int G1, const int M, const int N, const int K, const int O, const onnxruntime::DeviceArray &a_device_buf, const onnxruntime::DeviceArray &b0_device_buf, const onnxruntime::DeviceArray &b1_device_buf, const onnxruntime::DeviceArray &c_device_buf, const int best_op_id)

{
    // A layout [G0, M, G1, K]
    std::vector<ck::index_t> a_gs_ms_ks_lengths{G0, G1, M, K};
    //std::vector<ck::index_t> a_gs_ms_ks_strides{G1 * M * K, K, G1 * K, 1};
    std::vector<ck::index_t> a_gs_ms_ks_strides{G1 * M * K, M * K, K, 1};

    // B0 layout [G0, N, G1, K]
    std::vector<ck::index_t> b0_gs_ns_ks_lengths{G0, G1, N, K};
    //std::vector<ck::index_t> b0_gs_ns_ks_strides{G1 * N * K, K, G1 * K, 1};
    std::vector<ck::index_t> b0_gs_ns_ks_strides{G1 * N * K, N * K, K, 1};

    // B1 layout [G0, N, G1, O]
    std::vector<ck::index_t> b1_gs_os_ns_lengths{G0, G1, O, N};
    //std::vector<ck::index_t> b1_gs_os_ns_strides{G1 * N * O, O, 1, G1 * O};
    std::vector<ck::index_t> b1_gs_os_ns_strides{G1 * N * O, N * O, 1, O};

    // C layout [G0, M, G1, O]
    std::vector<ck::index_t> c_gs_ms_os_lengths{G0, G1, M, O};
    //std::vector<ck::index_t> c_gs_ms_os_strides{G1 * M * O, O, G1 * O, 1};
    std::vector<ck::index_t> c_gs_ms_os_strides{G1 * M * O, O, G1 * O, 1};

    auto a  =  a_device_buf.ptr();
    auto b0 = b0_device_buf.ptr();
    auto b1 = b1_device_buf.ptr();
    auto c  =  c_device_buf.ptr();

    using DeviceOp =
        ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute<2,
                                                                          1,
                                                                          1,
                                                                          1,
                                                                          1,
                                                                          ADataType,
                                                                          B0DataType,
                                                                          B1DataType,
                                                                          CDataType,
                                                                          ck::Tuple<>,
                                                                          ck::Tuple<>,
                                                                          AElementOp,
                                                                          B0ElementOp,
                                                                          Acc0ElementOp,
                                                                          B1ElementOp,
                                                                          CElementOp,
                                                                          MaskingSpec>;

    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    {
        auto& op_ptr = op_ptrs[best_op_id];
        auto argument_ptr = op_ptr->MakeArgumentPointer((ck::half_t*) a,
                                                        (ck::half_t*) b0,
                                                        (ck::half_t*) b1,
                                                        (ck::half_t*) c,
                                                        {}, // p_acc0_biases
                                                        {}, // p_acc1_biases
                                                        a_gs_ms_ks_lengths,
                                                        a_gs_ms_ks_strides,
                                                        b0_gs_ns_ks_lengths,
                                                        b0_gs_ns_ks_strides,
                                                        b1_gs_os_ns_lengths,
                                                        b1_gs_os_ns_strides,
                                                        c_gs_ms_os_lengths,
                                                        c_gs_ms_os_strides,
                                                        {}, // acc0_biases_gs_ms_ns_lengths
                                                        {}, // acc0_biases_gs_ms_ns_strides
                                                        {}, // acc1_biases_gs_ms_os_lengths
                                                        {}, // acc1_biases_gs_ms_os_strides
                                                        AElementOp{},
                                                        B0ElementOp{},
                                                        Acc0ElementOp{1 / sqrtf(K)},
                                                        B1ElementOp{},
                                                        CElementOp{});

        auto invoker_ptr = op_ptr->MakeInvokerPointer();

        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        }
    }

}
