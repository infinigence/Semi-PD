#pragma once
#include <cstdint>
#include <cinttypes>
#include <cuda_runtime.h>

namespace st::kernel {

template<typename T>
void addbias(
	T* output,
	const T* input,
	const T* bias,
	const int64_t size,
	cudaStream_t stream = nullptr
);

template<typename T>
void addbiasBatched(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch_size,
	const int64_t size,
	cudaStream_t stream = nullptr
);

}	// namespace st::kernel