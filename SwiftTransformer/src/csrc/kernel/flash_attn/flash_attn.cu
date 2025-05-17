#include "flash_attn.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <torch/torch.h>

#include "util/cuda_utils.h"
#include "util/debug_utils.h"

namespace st::kernel {

template<typename T>
void FlashAttention(
	T* __restrict__ result,
	const T* __restrict__ qkvs,
	const float qk_scale,
	const int32_t* __restrict__ input_lens,
	const int64_t num_context_reqs,
	const int64_t* __restrict__ ith_context_req_req_index,		// WARNING currently this kernel only support ith_context_req_req_index[i] == i
	const int32_t* __restrict__ ith_context_req_token_index,	// batch_size+1
	const int64_t num_q_heads,
	const int64_t num_kv_heads,
	const int64_t head_dim,
	const int64_t num_tokens,
	const int64_t max_context_req_len, 
	cudaStream_t stream
) {
	if constexpr (std::is_same_v<T, float>) {
		// Does not support float now!
		assert_whenever(0);
	} else {
		auto getTensor = [](void* data, torch::IntArrayRef sizes, const std::vector<int64_t> &dimension_strides, torch::ScalarType dtype, torch::Device device = torch::kCUDA) {
			const int64_t dim = dimension_strides.size();
			std::vector<int64_t> strides(dim);
			for (int64_t i = dim-1; i >= 0; --i) {
				int64_t last_dim_stride = i == dim-1 ? 1 : strides[i+1];
				int64_t last_dim_size = i == dim-1 ? 1 : sizes[i+1];
				strides[i] = last_dim_stride * last_dim_size * dimension_strides[i];
			}
			auto options = torch::TensorOptions().dtype(dtype).device(device);
			return torch::from_blob(data, sizes, strides, [](void*) {}, options);
		};
        auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
		// at::Tensor q_tensor = getTensor(
		// 	const_cast<T*>(qkvs),
		// 	{ num_tokens, num_heads, head_dim },
		// 	{ (num_heads == 32 ? 1.5 : 1.25), 1, 1 },
		// 	torch::kHalf
		// );
        at::Tensor q_tensor = torch::from_blob(
            // qkvs.data_ptr<at::Half>(), 
			const_cast<T*>(qkvs), 
            { num_tokens, num_q_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		// at::Tensor k_tensor = getTensor(
		// 	const_cast<T*>(qkvs) + num_heads * head_dim,
		// 	{ num_tokens, num_heads, head_dim },
		// 	{ 3, 1, 1 },
		// 	torch::kHalf
		// );
        at::Tensor k_tensor = torch::from_blob(
            // qkvs.data_ptr<at::Half>() + num_heads * head_dim, 
			const_cast<T*>(qkvs) + num_q_heads * head_dim, 
            { num_tokens, num_kv_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		// at::Tensor v_tensor = getTensor(
		// 	const_cast<T*>(qkvs) + 2 * num_heads * head_dim,
		// 	{ num_tokens, num_heads, head_dim },
		// 	{ 3, 1, 1 },
		// 	torch::kHalf
		// );
        at::Tensor v_tensor = torch::from_blob(
            // qkvs.data_ptr<at::Half>() + (8 + num_heads) * head_dim,
			const_cast<T*>(qkvs) + (num_q_heads + num_kv_heads) * head_dim, 
            { num_tokens, num_kv_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		at::Tensor seqstart = getTensor(
			const_cast<int32_t*>(ith_context_req_token_index),
			{ num_context_reqs + 1 },
			{ 1 },
			torch::kInt32
		);
		at::Tensor result_tensor = getTensor(
			result,
			{ num_tokens, num_q_heads, head_dim },
			{ 1, 1, 1 },
			torch::kHalf
		);
		c10::optional<at::Tensor> none = c10::nullopt;
		c10::optional<at::Tensor> result_tensor_cvt = result_tensor;
		mha_varlen_fwd(
			q_tensor,
			k_tensor,
			v_tensor,
            result_tensor_cvt,
			seqstart,
			seqstart,
            // c10::nullopt,
			none, 
            // c10::nullopt,
			none, 
            // c10::nullopt,
			none, 
			max_context_req_len,
            max_context_req_len,
			0.0,
            (double)qk_scale,
			false,
			true, 
			-1, 
            -1,
			false,
            c10::nullopt,
			stream
		);
		sync_check_cuda_error();
	}
}


template<typename T>
void FlashDecoding(
	T* __restrict__ result,
	const T* __restrict__ qkvs, // [num_tokens, num_q_heads+2*num_kv_heads, head_dim]
	T* k_cache,
	T* v_cache,
	const float qk_scale, 
	const int32_t* __restrict__ block_table,   // [num_reqs, max_num_block_per_seq]
	const int64_t num_blocks, 	// [hongke@1014] 
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
) {
	if constexpr (std::is_same_v<T, float>) {
		// Does not support float now!
		assert_whenever(0);
	} else {
		auto getTensor = [](void* data, torch::IntArrayRef sizes, const std::vector<int64_t> &dimension_strides, torch::ScalarType dtype, torch::Device device = torch::kCUDA) {
			const int64_t dim = dimension_strides.size();
			std::vector<int64_t> strides(dim);
			for (int64_t i = dim-1; i >= 0; --i) {
				int64_t last_dim_stride = i == dim-1 ? 1 : strides[i+1];
				int64_t last_dim_size = i == dim-1 ? 1 : sizes[i+1];
				strides[i] = last_dim_stride * last_dim_size * dimension_strides[i];
			}
			auto options = torch::TensorOptions().dtype(dtype).device(device);
			return torch::from_blob(data, sizes, strides, [](void*) {}, options);
		};
        auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
		// Q, K, V: [num_tokens, num_q_heads, head_dim]
        at::Tensor q_tensor = torch::from_blob(
			const_cast<T*>(qkvs), 
            { num_decoding_reqs, num_q_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		q_tensor = q_tensor.reshape({num_decoding_reqs, 1, num_q_heads, head_dim});
        at::Tensor k_tensor = torch::from_blob(
			const_cast<T*>(qkvs) + num_q_heads * head_dim, 
            { num_decoding_reqs, num_kv_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		k_tensor = k_tensor.reshape({num_decoding_reqs, 1, num_kv_heads, head_dim});
        at::Tensor v_tensor = torch::from_blob(
			const_cast<T*>(qkvs) + (num_q_heads + num_kv_heads) * head_dim, 
            { num_decoding_reqs, num_kv_heads, head_dim }, 
            { (num_q_heads + 2 * num_kv_heads) * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		v_tensor = v_tensor.reshape({num_decoding_reqs, 1, num_kv_heads, head_dim});
		// KVcache: [num_blocks, num_layers, local_kv_head_num, block_size, head_dim] -> [num_blocks, block_size, num_kv_heads, head_dim]
		at::Tensor k_cache_tensor = torch::from_blob(
			const_cast<T*>(k_cache) + layer_id * (num_kv_heads * block_size * head_dim), 
            { num_blocks, num_kv_heads, block_size, head_dim }, 
            { num_layers * num_kv_heads * block_size * head_dim, block_size * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		k_cache_tensor = k_cache_tensor.transpose(1, 2);
		at::Tensor v_cache_tensor = torch::from_blob(
			const_cast<T*>(v_cache) + layer_id * (num_kv_heads * block_size * head_dim), 
            { num_blocks, num_kv_heads, block_size, head_dim }, 
            { num_layers * num_kv_heads * block_size * head_dim, block_size * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		v_cache_tensor = v_cache_tensor.transpose(1, 2);	
		// Input len: [batch_size] 
		at::Tensor input_lens_tensor = getTensor(
			const_cast<int32_t*>(input_lens),
			{ num_decoding_reqs },
			{ 1 },
			torch::kInt32
		);
		// Block table: [num_reqs, max_num_block_per_seq]
		at::Tensor block_table_tensor = getTensor(
			const_cast<int32_t*>(block_table),
			{ num_decoding_reqs, max_num_block_per_seq },
			{ 1, 1 },
			torch::kInt32
		);
		// Output: [num_tokens, num_q_heads, head_dim]
		at::Tensor result_tensor = torch::from_blob(
			const_cast<T*>(result), 
            { num_decoding_reqs, num_q_heads, head_dim }, 
            { num_q_heads * head_dim, head_dim, 1 }, 
            [](void*) {}, 
            options);
		result_tensor = result_tensor.reshape({num_decoding_reqs, 1, num_q_heads, head_dim});
		// at::Tensor result_tensor = torch::empty({q_tensor.size(0), q_tensor.size(1), q_tensor.size(2), q_tensor.size(3)}, options);
		c10::optional<at::Tensor> none = c10::nullopt;
		c10::optional<const at::Tensor> cnone = c10::nullopt;
		c10::optional<at::Tensor> result_tensor_cvt = result_tensor;
		c10::optional<const at::Tensor> k_tensor_cvt = k_tensor;
		c10::optional<const at::Tensor> v_tensor_cvt = v_tensor;
		c10::optional<const at::Tensor> input_lens_tensor_cvt = input_lens_tensor;
		c10::optional<at::Tensor> block_table_tensor_cvt = block_table_tensor;
		mha_fwd_kvcache(
			q_tensor, // [num_tokens, 1, num_q_heads, head_dim]
			k_cache_tensor,  // [num_blocks, block_size, num_kv_heads, head_dim]
			v_cache_tensor,  // [num_blocks, block_size, num_kv_heads, head_dim]
			k_tensor_cvt, // [num_tokens, 1, num_kv_heads, head_dim]
			v_tensor_cvt, // [num_tokens, 1, num_kv_heads, head_dim]
			input_lens_tensor_cvt, // [batch_size] 
			cnone, 
			cnone,
			cnone, 
			block_table_tensor_cvt, // [num_reqs, max_num_block_per_seq]
			none, 
			result_tensor_cvt, 
            (float)qk_scale,
			true,
			-1, 
            -1,
			true,
            -1
		);
		sync_check_cuda_error();
	}
}


#define INSTANTIALIZE_FLASH_CONTEXT_STAGE_ATTENTION(T) \
	template void FlashAttention<T>( \
		T* __restrict__ result, \
		const T* __restrict__ qkvs,	\
		const float qk_scale, \
		const int32_t* __restrict__ input_lens, \
		const int64_t num_context_reqs, \
		const int64_t* __restrict__ ith_context_req_req_index, \
		const int32_t* __restrict__ ith_context_req_token_index, \
		const int64_t num_q_heads, \
		const int64_t num_kv_heads, \
		const int64_t head_dim, \
		const int64_t num_tokens, \
		const int64_t max_context_req_len, \
		cudaStream_t stream \
	);

INSTANTIALIZE_FLASH_CONTEXT_STAGE_ATTENTION(float)
INSTANTIALIZE_FLASH_CONTEXT_STAGE_ATTENTION(half)


// #define INSTANTIALIZE_FLASH_DECODING_STAGE_ATTENTION(T) \
// 	template void FlashDecoding<T>( \
// 		T* __restrict__ result, \
// 		const T* __restrict__ qkvs, \
// 		T* k_cache, \
// 		T* v_cache, \
// 		const float scale, \
// 		const int64_t* __restrict__ block_table, \
// 		const int64_t num_blocks, \
// 		const int64_t* __restrict__ input_lens, \
// 		const int64_t num_decoding_reqs, \
// 		const int64_t* __restrict__ ith_decoding_req_req_index, \
// 		const int64_t* __restrict__ ith_decoding_req_token_index, \
// 		const int64_t max_decoding_req_len, \
// 		const int64_t num_layers, \
// 		const int64_t num_q_heads, \
// 		const int64_t num_kv_heads, \
// 		const int64_t head_dim, \
// 		const int64_t layer_id, \
// 		const int64_t block_size, \
// 		const int64_t max_num_block_per_seq \
// 	);

// INSTANTIALIZE_FLASH_DECODING_STAGE_ATTENTION(float)
// INSTANTIALIZE_FLASH_DECODING_STAGE_ATTENTION(half)

template void FlashDecoding(
	half* __restrict__ result,
	const half* __restrict__ qkvs,
	half* k_cache,
	half* v_cache,
	const float scale,
	const int32_t* __restrict__ block_table,
	const int64_t num_blocks, 
	const int32_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
	const int64_t max_decoding_req_len,
	const int64_t num_layers,
	const int64_t num_q_heads,
	const int64_t num_kv_heads, 
	const int64_t head_dim,
	const int64_t layer_id,
	const int64_t block_size,
	const int64_t max_num_block_per_seq
);
template void FlashDecoding(
	float* __restrict__ result,
	const float* __restrict__ qkvs,
	float* k_cache,
	float* v_cache,
	const float scale,
	const int32_t* __restrict__ block_table,
	const int64_t num_blocks, 
	const int32_t* __restrict__ input_lens,
	const int64_t num_decoding_reqs,
	const int64_t* __restrict__ ith_decoding_req_req_index,
	const int64_t* __restrict__ ith_decoding_req_token_index,
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