#include "rotary_posi_embedding.h"
#include <math_constants.h>
namespace st::kernel {

/*
	rotaryPosiEmbeddingBatched

	Perform rotary positional embedding on a batch of tokens.

	## Background

	Rotary positional embedding (RoPE), as proposed in "ROFORMER : ENHANCED
	TRANSFORMER WITH ROTARY POSITION EMBEDDING", is a method of positional
	embedding that encodes the absolute position while incorporates the
	relative position between tokens. Models like LLaMA and LLaMA2 are based
	on RoPE.

	## Introduction

	This kernel takes a bunch of tokens and their absolute positions with in the
	request, and performs RoPE on them.

	## Implementation Details

	We summon a grid of shape (num_tokens), i.e. each thread block is
	responsible for one token. Each thread block has head_dim/2 threads. The 
	i-th thread will deal with the (2i) and (2i+1) th elements in the head_dim
	in every head.

	## Notes

	In practice we perform RoPE on both the query and the key. Note that when
	performing RoPE on the key, we need to pass num_local_kv_heads as num_heads,
	while performing on the query we need to pass num_local_q_heads as num_heads.
*/

template<typename T>
__global__ void rotaryPosiEmbeddingBatchedKernel (
	T* __restrict__ target,		// [num_tokens, target_1st_dim_size, head_dim]. We will only use [num_tokens, :num_heads, head_dim]
	const int64_t* __restrict__ token_indexes,	// [num_tokens]
	const int64_t num_heads,
	const int64_t target_1st_dim_size,
	const int64_t head_dim
) {
	const int64_t rel_pos = token_indexes[blockIdx.x];
	float cur_sin_f_0, cur_cos_f_0;
	__sincosf(rel_pos*__powf(500000.0f, -2.0f*threadIdx.x * 2/head_dim), &cur_sin_f_0, &cur_cos_f_0);
	float cur_sin_f_1, cur_cos_f_1;
	__sincosf(rel_pos*__powf(500000.0f, -2.0f*(threadIdx.x * 2 + 1)/head_dim), &cur_sin_f_1, &cur_cos_f_1);

	const T cur_sin_0 = (T)cur_sin_f_0, cur_cos_0 = (T)cur_cos_f_0;
	const T cur_sin_1 = (T)cur_sin_f_1, cur_cos_1 = (T)cur_cos_f_1;

	typedef typename std::conditional<std::is_same<T, float>::value, float2, half2>::type T2;
	for (int64_t head_id = 0; head_id < num_heads; head_id += 1) {
		// Read x1 and x2 in pack
		// const T2 x1_x2 = reinterpret_cast<T2*>(target)[INDEX_3D(
		// 	0, target_1st_dim_size, head_dim/2,
		// 	blockIdx.x, head_id, threadIdx.x
		// )];
		// const T x1 = x1_x2.x, x2 = x1_x2.y;

		// int64_t head_id = threadIdx.y;
		int64_t stride_token = target_1st_dim_size * head_dim / 2;
		const T2 real_x1_x2 = reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x];
		const T2 complex_x1_x2 = reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x + head_dim / 4];


		// const T2 real_x1_x2 = reinterpret_cast<T2*>(target)[INDEX_3D(
		// 	0, target_1st_dim_size, head_dim/2,
		// 	blockIdx.x, head_id, threadIdx.x
		// )];
		// const T2 complex_x1_x2 = reinterpret_cast<T2*>(target)[INDEX_3D(
		// 	0, target_1st_dim_size, head_dim/2,
		// 	blockIdx.x, head_id, threadIdx.x + head_dim / 4
		// )];
		
		const T real_x1 = real_x1_x2.x;
		const T real_x2 = real_x1_x2.y;
		const T complex_x1 = complex_x1_x2.x;
		const T complex_x2 = complex_x1_x2.y;

		const T new_real_x1 = real_x1 * cur_cos_0 - complex_x1 * cur_sin_0;
		const T new_complex_1 = real_x1 * cur_sin_0 + complex_x1 * cur_cos_0;

		const T new_real_x2 = real_x2 * cur_cos_1 - complex_x2 * cur_sin_1;
		const T new_complex_2 = real_x2 * cur_sin_1 + complex_x2 * cur_cos_1;


		// float real_x1_f = __half2float(real_x1);
		// float complex_x1_f = __half2float(complex_x1_x2.x);
		// float new_real_x1_f = __half2float(new_real_x1);

		// float real_x2_f = __half2float(real_x2);
		// float complex_x2_f = __half2float(complex_x2);
		// float new_real_x2_f = __half2float(new_real_x2);
		// if (threadIdx.x ==0 && blockIdx.x ==262 &&head_id ==0)
		// {
		// 	printf("stride_token: %ld real_x1_f: %f, complex_x1_f: %f, new_real_x1_f: %f\n ",stride_token, real_x1_f, complex_x1_f, new_real_x1_f);
		// 	printf("stride_token: %ld real_x2_f: %f, complex_x2_f: %f, new_real_x2_f: %f\n ",stride_token, real_x2_f, complex_x2_f, new_real_x2_f);
		// }
		
		

		// const T new_x1 = x1*cur_cos - x2*cur_sin;
		// const T new_x2 = x1*cur_sin + x2*cur_cos;
		// Write back
		// reinterpret_cast<T2*>(target)[INDEX_3D(
		// 	0, target_1st_dim_size, head_dim/2,
		// 	blockIdx.x, head_id, threadIdx.x
		// )] = T2{new_real_x1, new_real_x2};

		// reinterpret_cast<T2*>(target)[INDEX_3D(
		// 	0, target_1st_dim_size, head_dim/2,
		// 	blockIdx.x, head_id, threadIdx.x + head_dim / 4
		// )] = T2{new_complex_1, new_complex_2};
		reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x] = T2{new_real_x1, new_real_x2};
		reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x + head_dim / 4]  = T2{new_complex_1, new_complex_2};


	}
}

template<typename T>
void rotaryPosiEmbeddingBatched(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim,
	cudaStream_t stream
) {
	// dim3 blockdim;
	// blockdim.x = head_dim / 4;
	// blockdim.y = num_heads;
	rotaryPosiEmbeddingBatchedKernel<T><<<num_tokens, {head_dim/4}, 0, stream>>>(
	// rotaryPosiEmbeddingBatchedKernel<T><<<num_tokens, blockdim, 0, stream>>>(
		target, token_indices, num_heads, target_1st_dim_size, head_dim
	);
}

#define INTANTIATE(T) \
	template void rotaryPosiEmbeddingBatched<T>( \
		T* __restrict__, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		cudaStream_t \
	);

INTANTIATE(half)
INTANTIATE(float)

template<typename T>
__global__ void llama3rotaryPosiEmbeddingBatchedKernel (
	T* __restrict__ target,		// [num_tokens, target_1st_dim_size, head_dim]. We will only use [num_tokens, :num_heads, head_dim]
	const int64_t* __restrict__ token_indexes,	// [num_tokens]
	const int64_t num_heads,
	const int64_t target_1st_dim_size,
	const int64_t head_dim
) {
	const int64_t rel_pos = token_indexes[blockIdx.x];
	float cur_sin_f, cur_cos_f;
	float freq = __powf(10000.0f, -2.0f*threadIdx.x/head_dim);
	float wave_len = M_PI * 2 / freq;
	if (wave_len < 1024){
		__sincosf(rel_pos*freq, &cur_sin_f, &cur_cos_f);
	};
	if (wave_len > 8192){
		__sincosf(rel_pos / 8 *freq, &cur_sin_f, &cur_cos_f);
	} else if ( wave_len >= 1024 && wave_len <= 8192)
	{
		float smooth = (8192 / wave_len - 1) / 3;
		float smoothed_freq = (1 - smooth) * freq / 8 + smooth * freq;
		__sincosf(rel_pos / smoothed_freq, &cur_sin_f, &cur_cos_f);
	}
	

	const T cur_sin = (T)cur_sin_f, cur_cos = (T)cur_cos_f;

	typedef typename std::conditional<std::is_same<T, float>::value, float2, half2>::type T2;
	for (int64_t head_id = 0; head_id < num_heads; head_id += 1) {
		// Read x1 and x2 in pack
		const T2 x1_x2 = reinterpret_cast<T2*>(target)[INDEX_3D(
			0, target_1st_dim_size, head_dim/2,
			blockIdx.x, head_id, threadIdx.x
		)];
		const T x1 = x1_x2.x, x2 = x1_x2.y;
		const T new_x1 = x1*cur_cos - x2*cur_sin;
		const T new_x2 = x1*cur_sin + x2*cur_cos;
		// Write back
		reinterpret_cast<T2*>(target)[INDEX_3D(
			0, target_1st_dim_size, head_dim/2,
			blockIdx.x, head_id, threadIdx.x
		)] = T2{new_x1, new_x2};
	}
}

template<typename T>
void llama3rotaryPosiEmbeddingBatched(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim,
	cudaStream_t stream
) {
	llama3rotaryPosiEmbeddingBatchedKernel<T><<<num_tokens, head_dim/2, 0, stream>>>(
		target, token_indices, num_heads, target_1st_dim_size, head_dim
	);
}

#define INTANTIATE_LLAMA3(T) \
	template void llama3rotaryPosiEmbeddingBatched<T>( \
		T* __restrict__, \
		const int64_t* __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		cudaStream_t \
	);

INTANTIATE_LLAMA3(half)
INTANTIATE_LLAMA3(float)


template<typename T>
__global__ void rotaryPosiEmbeddingBatchedKernelWithCosSinCache (
	T* __restrict__ target,		// [num_tokens, target_1st_dim_size, head_dim]. We will only use [num_tokens, :num_heads, head_dim]
	const int64_t* __restrict__ token_indexes,	// [num_tokens]
	// const float*  __restrict__ cos_sin_cache,	// [head_dim]
	const half*  __restrict__ cos_sin_cache,	// [head_dim]
	const int64_t num_heads,
	const int64_t target_1st_dim_size,
	const int64_t head_dim
) {
	const int64_t rel_pos = token_indexes[blockIdx.x];

	int2 cos_sin_x1x2_y1y2 = ((int2*)(cos_sin_cache))[rel_pos * 32 + threadIdx.x];
	half2 cos_x1x2 = *((half2*)(&cos_sin_x1x2_y1y2.x));
	half2 sin_x1x2 = *((half2*)(&cos_sin_x1x2_y1y2.y));
	const T cur_cos_0 = (T)cos_x1x2.x;
	const T cur_cos_1 = (T)cos_x1x2.y;
	const T cur_sin_0 = (T)sin_x1x2.x;
	const T cur_sin_1 = (T)sin_x1x2.y;

	typedef typename std::conditional<std::is_same<T, float>::value, float2, half2>::type T2;
	for (int64_t head_id = 0; head_id < num_heads; head_id += 1) {
		int64_t stride_token = target_1st_dim_size * head_dim / 2;
		const T2 real_x1_x2 = reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x];
		const T2 complex_x1_x2 = reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x + head_dim / 4];

		
		const T real_x1 = real_x1_x2.x;
		const T real_x2 = real_x1_x2.y;
		const T complex_x1 = complex_x1_x2.x;
		const T complex_x2 = complex_x1_x2.y;

		const T new_real_x1 = real_x1 * cur_cos_0 - complex_x1 * cur_sin_0;
		const T new_complex_1 = real_x1 * cur_sin_0 + complex_x1 * cur_cos_0;

		const T new_real_x2 = real_x2 * cur_cos_1 - complex_x2 * cur_sin_1;
		const T new_complex_2 = real_x2 * cur_sin_1 + complex_x2 * cur_cos_1;

		reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x] = T2{new_real_x1, new_real_x2};
		reinterpret_cast<T2*>(target)[stride_token * blockIdx.x + head_id * head_dim / 2 + threadIdx.x + head_dim / 4]  = T2{new_complex_1, new_complex_2};


	}
}

template<typename T>
void rotaryPosiEmbeddingBatchedWithCosSinCache(
	T* __restrict__ target,
	const int64_t* __restrict__ token_indices,
	const half*  __restrict__ cos_sin_cache,	// [head_dim]
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim,
	cudaStream_t stream
) {
	rotaryPosiEmbeddingBatchedKernelWithCosSinCache<T><<<num_tokens, head_dim/4, 0, stream>>>(
		target, token_indices, cos_sin_cache, num_heads, target_1st_dim_size, head_dim
	);
}

#define INTANTIATE_WITH_COSSIN(T) \
	template void rotaryPosiEmbeddingBatchedWithCosSinCache<T>( \
		T* __restrict__, \
		const int64_t* __restrict__, \
		const half*  __restrict__, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		const int64_t, \
		cudaStream_t \
	);

INTANTIATE_WITH_COSSIN(half)
INTANTIATE_WITH_COSSIN(float)


}	// namespace st::kernel
