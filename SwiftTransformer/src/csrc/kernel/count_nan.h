#pragma once

namespace st::kernel {

template<typename T>
int countNan(
	const T* arr,
	int n,
	cudaStream_t stream = nullptr
);

}