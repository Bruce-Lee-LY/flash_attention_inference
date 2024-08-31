// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v2.1.0

#include "flash_attn_v2.h"

#include "cutlass/half.h"
#include "flash_attn_v2/flash.h"
#include "flash_attn_v2/static_switch.h"
#include "tensor.h"

#define FAI_M_LOG2E 1.4426950408889634074  // log_2 e

Flash_fwd_params set_mha_fwd_params(void *q, void *k, void *v, void *o, int *cu_seq_q, int *cu_seq_k, size_t max_seq_q,
                                    size_t max_seq_k, size_t batch, size_t head_q, size_t head_k, size_t dim,
                                    bool is_causal, bool is_alibi, cudaDeviceProp *dev_prop) {
    FAI_CHECK(q);
    FAI_CHECK(k);
    FAI_CHECK(v);
    FAI_CHECK(o);
    FAI_CHECK(cu_seq_q);
    FAI_CHECK(cu_seq_k);
    FAI_CHECK_EQ(head_q % head_k, 0);
    FAI_CHECK_LE(dim, 256);

    Flash_fwd_params params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;

    // Calculate batch_stride using cu_seq
    params.q_batch_stride = 0;
    params.k_batch_stride = 0;
    params.v_batch_stride = 0;
    params.q_row_stride = head_q * dim;
    params.k_row_stride = head_k * dim;
    params.v_row_stride = head_k * dim;
    params.q_head_stride = dim;
    params.k_head_stride = dim;
    params.v_head_stride = dim;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = o;

    // Calculate batch_stride using cu_seq
    params.o_batch_stride = 0;
    params.o_row_stride = head_q * dim;
    params.o_head_stride = dim;

    // Softmax sum
    Tensor<float> *softmax_lse = new Tensor<float>({batch, head_q, max_seq_q}, "Tensor softmax_lse");
    FAI_CHECK(softmax_lse);
    params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = max_seq_q;
    params.seqlen_k = max_seq_k;
    params.d = dim;

    params.scale_softmax = 1.0 / std::sqrt(dim);
    params.scale_softmax_log2 = params.scale_softmax * FAI_M_LOG2E;

    params.cu_seqlens_q = cu_seq_q;
    params.cu_seqlens_k = cu_seq_k;

    params.is_causal = is_causal;
    params.is_alibi = is_alibi;

    params.props = dev_prop;
    params.is_sm8x = params.props->major == 8 && params.props->minor > 0;

    return params;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FWD_HEADDIM_SWITCH(params.d, [&] { run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream); });
}

void flash_attn_v2(void *q, void *k, void *v, void *o, int *cu_seq_q, int *cu_seq_k, size_t total_q, size_t max_seq_q,
                   size_t max_seq_k, size_t batch, size_t head_q, size_t head_k, size_t dim, bool is_causal,
                   int num_splits, bool is_alibi, cudaStream_t stream, cudaDeviceProp *dev_prop) {
    FAI_UNUSED(total_q);
    FAI_UNUSED(num_splits);
    static Flash_fwd_params params = set_mha_fwd_params(q, k, v, o, cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, batch,
                                                        head_q, head_k, dim, is_causal, is_alibi, dev_prop);
    run_mha_fwd(params, stream);
}
