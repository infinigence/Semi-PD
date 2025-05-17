#pragma once

#include "util/cuda_utils.h"

namespace st::kernel {

template<typename T>
void rotaryPosiEmbeddingBatched(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim,
	cudaStream_t stream = nullptr
);


template<typename T>
void rotaryPosiEmbeddingBatchedWithCosSinCache(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	// const float* __restrict__ cos_sin_cache,
	const half* __restrict__ cos_sin_cache,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim,
	cudaStream_t stream = nullptr
);

}	// namespace st::kernel
