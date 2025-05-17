#pragma once

#include <torch/extension.h>
#include <vector>

namespace st::util {

void forwardRope(
	torch::Tensor &input,
    torch::Tensor &positions,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim
);

void forwardRopeWithCosSinCache(
	torch::Tensor &input,
    torch::Tensor &positions,
    torch::Tensor &cos_sin_cache,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim
);

} // namespace st::util