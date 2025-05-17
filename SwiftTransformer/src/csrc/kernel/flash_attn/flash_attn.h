#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

std::vector<at::Tensor> mha_varlen_fwd(at::Tensor &q,
                                       // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                                       const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k :=
                                                             // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                                                             // x num_heads_k x head_size if there's a block_table.
                                       const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k :=
                                                             // \sum_{i=0}^{b} s_i or num_blocks x page_block_size
                                                             // x num_heads_k x head_size if there's a block_table.
                                       c10::optional<at::Tensor> &out_,
                                       // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                                       const at::Tensor &cu_seqlens_q,  // b+1
                                       const at::Tensor &cu_seqlens_k,  // b+1
                                       c10::optional<at::Tensor> &seqused_k,
                                       // b. If given, only this many elements of each batch element's keys are
                                       // used.
                                       c10::optional<at::Tensor> &block_table_,
                                       // batch_size x max_num_blocks_per_seq
                                       c10::optional<at::Tensor> &alibi_slopes_,
                                       // num_heads or b x num_heads
                                       int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
                                       const float softmax_scale, const bool zero_tensors, bool is_causal,
                                       int window_size_left, int window_size_right, const bool return_softmax,
                                       c10::optional<at::Generator> gen_,
                                       cudaStream_t stream);
									   


std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor &q,                                // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                     // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x
                                                  // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor &vcache,                     // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x
                                                  // page_block_size x num_heads_k x head_size if there's a block_table.
    c10::optional<const at::Tensor> &k_,          // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &v_,          // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &seqlens_k_,  // batch_size
    c10::optional<const at::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
    c10::optional<at::Tensor> &block_table_,            // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_,           // num_heads or batch_size x num_heads
    c10::optional<at::Tensor> &out_,                    // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 &
                                 // rotary_dim / 2
    int num_splits);

namespace st::kernel {

template<typename T>
void FlashAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	const float qk_scale,
	const int32_t* __restrict__ input_lens,
	const int64_t num_context_reqs,
	const int64_t* __restrict__ ith_context_req_req_index,
	const int32_t* __restrict__ ith_context_req_token_index,
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t num_tokens,
	const int64_t max_context_req_len,
    cudaStream_t stream
);

template<typename T>
void FlashDecoding(
	T* __restrict__ result,
	const T* __restrict__ qkvs, // [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	T* k_cache,
	T* v_cache,
	const float scale, 
	const int32_t* __restrict__ block_table,   // [num_reqs, max_num_block_per_seq]
    const int64_t num_blocks, 
	const int32_t* __restrict__ input_lens,    // [num_reqs]. Here input_lens DOES NOT INCLUDE the latest token!
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,  	// [num_decoding_reqs]
	const int64_t* __restrict__ ith_decoding_req_token_index,   // [num_decoding_reqs]
	const int64_t max_decoding_req_len, 
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads, 
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);

}