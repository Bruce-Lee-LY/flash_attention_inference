// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: flash attn v1.0.9

#include <cstddef>

#include "cuda_runtime_api.h"

/**
 * @brief flash attn api
 *
 * @param q [total_q * head_q * dim]
 * @param k [total_k * head_k * dim]
 * @param v [total_k * head_k * dim]
 * @param o [total_q * head_q * dim]
 * @param cu_seq_q [batch + 1]
 * @param cu_seq_k [batch + 1]
 * @param total_q
 * @param max_seq_q
 * @param max_seq_k
 * @param batch
 * @param head_q
 * @param head_k
 * @param dim
 * @param is_causal
 * @param num_splits
 * @param is_alibi
 * @param stream
 * @param dev_prop
 */
void flash_attn(void *q, void *k, void *v, void *o, int *cu_seq_q, int *cu_seq_k, size_t total_q, size_t max_seq_q,
                size_t max_seq_k, size_t batch, size_t head_q, size_t head_k, size_t dim, bool is_causal,
                int num_splits, bool is_alibi, cudaStream_t stream, cudaDeviceProp *dev_prop);
