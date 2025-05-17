#pragma once

#include <cstdint>
#include <cinttypes>

#include "activation_types.h"
#include "cuda_runtime.h"

namespace st::kernel {

template<typename T>
void fusedAddbiasBatchedActivation(
	T* output,
	const T* input,
	const T* bias,
	const int64_t batch,
	const int64_t size,
	ActivationType activation_type,
	cudaStream_t stream = nullptr
);

}	// namespace st::kernel