// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v1.0.9

#include "flash_attn.h"

#include "flash_attn/fmha.h"
#include "flash_attn/static_switch.h"
#include "tensor.h"

Launch_params<FMHA_fprop_params> set_fmha_fwd_params(void *q, void *k, void *v, void *o, int *cu_seq_q, int *cu_seq_k,
                                                     size_t total_q, size_t max_seq_q, size_t max_seq_k, size_t batch,
                                                     size_t head_q, size_t head_k, size_t dim, bool is_causal,
                                                     int num_splits, bool is_alibi, cudaStream_t stream,
                                                     cudaDeviceProp *dev_prop) {
    FAI_CHECK(q);
    FAI_CHECK(k);
    FAI_CHECK(v);
    FAI_CHECK(o);
    FAI_CHECK(cu_seq_q);
    FAI_CHECK(cu_seq_k);
    FAI_CHECK_EQ(head_q % head_k, 0);
    FAI_CHECK_LE(dim, 128);
    FAI_CHECK_EQ(dim % 8, 0);

    Launch_params<FMHA_fprop_params> launch_params(dev_prop, stream);
    FMHA_fprop_params &params = launch_params.params;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v;

    params.q_row_stride_in_elts = head_q * dim;
    params.k_row_stride_in_elts = head_k * dim;
    params.v_row_stride_in_elts = head_k * dim;
    params.q_head_stride_in_elts = dim;
    params.k_head_stride_in_elts = dim;
    params.v_head_stride_in_elts = dim;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = o;

    params.o_row_stride_in_elts = head_q * dim;
    params.o_head_stride_in_elts = dim;
    params.o_tmp_row_stride_in_elts = head_q * dim;
    params.o_tmp_head_stride_in_elts = dim;

    int blocksize_c = dim > 64 ? 128 : 256;
    // Need to round max_seqlen_k to multiples of blocksize_c
    int round_max_seq_k = ((max_seq_k + blocksize_c - 1) / blocksize_c) * blocksize_c;
    if (round_max_seq_k <= 128) {
        round_max_seq_k = 128;
    } else if (round_max_seq_k <= 256) {
        round_max_seq_k = 256;
    }

    if (round_max_seq_k > blocksize_c) {
        Tensor<float> *o_tmp = new Tensor<float>({total_q, head_q, dim}, "Tensor o_tmp");
        FAI_CHECK(o_tmp);
        params.o_tmp_ptr = reinterpret_cast<void *>(o_tmp->getDevPtr());
    } else {
        params.o_tmp_ptr = nullptr;
    }

    int round_max_seq_q = ((max_seq_q + 16 - 1) / 16) * 16;

    // Softmax sum
    Tensor<float> *softmax_lse =
        new Tensor<float>({batch, head_q, static_cast<size_t>(round_max_seq_q)}, "Tensor softmax_lse");
    FAI_CHECK(softmax_lse);
    params.softmax_lse_ptr = reinterpret_cast<void *>(softmax_lse->getDevPtr());

    // Set the dimensions.
    params.b = batch;
    params.seqlen_q = round_max_seq_q;
    params.seqlen_k = round_max_seq_k;
    params.d = dim;

    params.scale_bmm1f = 1.0 / std::sqrt(dim);
    set_alpha(params.scale_bmm1, params.scale_bmm1f, DATA_TYPE_FP16);

    params.cu_seqlens_q = cu_seq_q;
    params.cu_seqlens_k = cu_seq_k;

    params.is_causal = is_causal;

    params.num_splits = num_splits;

    params.is_alibi = is_alibi;

    return launch_params;
}

void run_fmha_fwd(Launch_params<FMHA_fprop_params> &launch_params) {
    if (launch_params.params.d <= 32) {
        run_fmha_fwd_hdim32(launch_params);
    } else if (launch_params.params.d <= 64) {
        run_fmha_fwd_hdim64(launch_params);
    } else if (launch_params.params.d <= 128) {
        run_fmha_fwd_hdim128(launch_params);
    }
}

void flash_attn(void *q, void *k, void *v, void *o, int *cu_seq_q, int *cu_seq_k, size_t total_q, size_t max_seq_q,
                size_t max_seq_k, size_t batch, size_t head_q, size_t head_k, size_t dim, bool is_causal,
                int num_splits, bool is_alibi, cudaStream_t stream, cudaDeviceProp *dev_prop) {
    static Launch_params<FMHA_fprop_params> launch_params =
        set_fmha_fwd_params(q, k, v, o, cu_seq_q, cu_seq_k, total_q, max_seq_q, max_seq_k, batch, head_q, head_k, dim,
                            is_causal, num_splits, is_alibi, stream, dev_prop);
    run_fmha_fwd(launch_params);
}
