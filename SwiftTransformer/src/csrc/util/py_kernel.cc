#include "py_kernel.h"
#include "util/torch_utils.h"
#include "kernel/rotary_posi_embedding.h"
namespace st::util {


// template<typename T>
void forwardRope(
	torch::Tensor &input,
    torch::Tensor &positions,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim
){
    std::cerr << "invoke rope" << std::endl;
	std::cerr << "args: num_tokens " << num_tokens << " target_1st_dim_size " << target_1st_dim_size \
	  << " num_heads " << num_heads;
	
	// st::kernel::rotaryPosiEmbeddingBatched<half>(
	// 	(half*)convertTensorToRawPtr(input),
	st::kernel::rotaryPosiEmbeddingBatched<half>(
		(half*)convertTensorToRawPtr(input),
		(int64_t*)(positions.data_ptr<int64_t>()),
		num_tokens,
		target_1st_dim_size,
		num_heads,
		head_dim
	);
}

void forwardRopeWithCosSinCache(
	torch::Tensor &input,
    torch::Tensor &positions,
    torch::Tensor &cos_sin_cache,
	const int64_t num_tokens,
	const int64_t target_1st_dim_size,
	const int64_t num_heads,
	const int64_t head_dim
){
    std::cerr << "invoke rope" << std::endl;
	std::cerr << "args: num_tokens " << num_tokens << " target_1st_dim_size " << target_1st_dim_size \
	  << " num_heads " << num_heads;
	
	// st::kernel::rotaryPosiEmbeddingBatched<half>(
	// 	(half*)convertTensorToRawPtr(input),
	st::kernel::rotaryPosiEmbeddingBatchedWithCosSinCache<half>(
		(half*)convertTensorToRawPtr(input),
		(int64_t*)(positions.data_ptr<int64_t>()),
        // (float*)(cos_sin_cache.data_ptr<float>()),
        (half*)(cos_sin_cache.data_ptr<at::Half>()),
		num_tokens,
		target_1st_dim_size,
		num_heads,
		head_dim
	);
}

} // namespace st::util