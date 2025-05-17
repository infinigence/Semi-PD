#include <filesystem>
#include <vector>

#include "gpt_weight.h"
#include "util/cuda_utils.h"
#include "util/torch_utils.h"
#include "util/py_block_migration.h"
#define TO_STRING(x) #x

namespace st::model {

template<typename T>
GptWeight<T>::GptWeight(){}

template<typename T>
void GptWeight<T>::init(const GptHyperParam& hyper_param, GptParallelismParam& parallelism_param) {
	this->hyper_param = hyper_param;
	this->parallelism_param = parallelism_param;
	this->parallelism_param.init_by_hyper_param(this->hyper_param);
	layer_weights.resize(this->parallelism_param.local_layer_num);
	contain_embedding_layer = this->parallelism_param.is_first_stage() || this->parallelism_param.is_last_stage();
	initialized = true;
	
	// allocateArray(cos_sin_cache, (hyper_param.head_dim + 127) / 128 * 128);

	// if (this->parallelism_param.is_context == 1){
	// 	// fprintf(stderr, "Is context==1");
	// 	AllocateWeightArrayForMPS();
	// }
	// AllocateWeightArrayForMPS(true);
}

template<typename T>
GptWeight<T>::~GptWeight() {
	if (this->parallelism_param.is_context == 1){
	freeWeightArray();
}
}

template<typename T>
void GptWeight<T>::LazyInitWeight(bool enable_ipc_mem, bool is_alloc) {
	if (enable_ipc_mem){
		AllocateWeightArrayForMPS(is_alloc);
		return;
	}
	// allocateWeightArrayFSDP();
	allocateWeightArray();
}


template<typename T>
void GptWeight<T>::AllocateWeightArrayForMPS(bool is_alloc) {
	int64_t index = 0;
	#define allocateArray(ptr, num_elem) \
		{ \
			if (is_alloc) { \
			CUDA_CHECK(cudaMalloc(&ptr, (num_elem)*sizeof(T))); \
			fprintf(stderr, "alloc weight tensor %s ipc ptr %ld \n", TO_STRING(ptr), index); \
			auto tmp_tensor = torch::from_blob(ptr, num_elem, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA)); \
			stage_flatten_tensor_vec_.push_back(tmp_tensor); \
			} else { \
				(ptr) = (static_cast<T*>(st::util::get_ipc_weight_ptr(index))); \
				fprintf(stderr, "decode instance assign weight tensor %s ipc ptr %ld \n", TO_STRING(ptr), index); \
				index ++; \
			} \
		}


	const int64_t local_num_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	const int64_t local_ffn_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	// get free gpu memory
	size_t free_byte;
	size_t total_byte;
	CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

	allocateArray(cos_sin_cache, hyper_param.max_position_embeddings * hyper_param.head_dim);



	// auto embed_tokens_weight_size = contain_embedding_layer ? hyper_param.vocab_size*hyper_param.hidden_size : 0;
	// auto embed_positions_weight_size = contain_embedding_layer && !hyper_param.is_rotary_posi_embedding ? hyper_param.max_position_embeddings*hyper_param.hidden_size : 0;
	// auto qkv_size = hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim;
	// auto qkv_bias_size = hyper_param.is_attn_qkv_biased ? (local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim : 0;
	// auto o_size = local_num_q_heads * hyper_param.head_dim*hyper_param.hidden_size;
	// auto o_bias_size = hyper_param.is_attn_out_biased ? hyper_param.hidden_size : 0;
	// auto attn_layernorm_weight_size =  hyper_param.hidden_size;
	// auto attn_layernorm_bias_size = !hyper_param.is_rmsnorm ? hyper_param.hidden_size : 0;
	// auto ffn_size = !hyper_param.is_gated_ffn ? local_ffn_dim * hyper_param.hidden_size * 2 : local_ffn_dim * hyper_param.hidden_size * 3;
	// auto ffn_bias_size = !hyper_param.is_gated_ffn ? local_ffn_dim + hyper_param.hidden_size : 0;
	// auto final_layernorm_weight_size = hyper_param.hidden_size;
	// auto final_layernorm_bias_size = !hyper_param.is_rmsnorm ? hyper_param.hidden_size : 0;
	// auto last_final_layernorm_weight_size = parallelism_param.is_last_stage() ? hyper_param.hidden_size :0;
	// auto last_final_layernorm_bias_size = !hyper_param.is_rmsnorm && parallelism_param.is_last_stage() ? hyper_param.hidden_size :0;
	// auto output_proj_weight_size = parallelism_param.is_last_stage() ? hyper_param.vocab_size*hyper_param.hidden_size : 0;


	// Allocate weights for embedding layer
	if (contain_embedding_layer) {
		allocateArray(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding) {
			allocateArray(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
		} else {
			embed_positions_weight = nullptr;
		}
	}
	else{
		embed_tokens_weight = nullptr;
		embed_positions_weight = nullptr;
	}

	// Allocate weights for each layer
	layer_weights.resize(parallelism_param.local_layer_num);
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		// GptLayerWeight<T>& layer_weight = layer_weights[i];

		allocateArray(layer_weights[i].attn_qkv_kernel, hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			allocateArray(layer_weights[i].attn_qkv_bias, (local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		}
		allocateArray(layer_weights[i].attn_out_kernel, local_num_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		if (hyper_param.is_attn_out_biased) {
			allocateArray(layer_weights[i].attn_out_bias, hyper_param.hidden_size);
		}

		allocateArray(layer_weights[i].attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weights[i].attn_layernorm_bias, hyper_param.hidden_size);
		}

		// allocateArray(layer_weights[i].ffn_fc1_weight, local_ffn_dim*hyper_param.hidden_size);
		allocateArray(layer_weights[i].ffn_fc2_weight, hyper_param.hidden_size*local_ffn_dim);
		if (!hyper_param.is_gated_ffn) {
			allocateArray(layer_weights[i].ffn_fc1_bias, local_ffn_dim);
			allocateArray(layer_weights[i].ffn_fc2_bias, hyper_param.hidden_size);
		} else {
			// allocateArray(layer_weights[i].ffn_fc3_weight, local_ffn_dim*hyper_param.hidden_size);
			allocateArray(layer_weights[i].ffn_fc13_weight, 2 * local_ffn_dim*hyper_param.hidden_size);
		}

		allocateArray(layer_weights[i].final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weights[i].final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Allocate weights for final layer
		allocateArray(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(final_layernorm_bias, hyper_param.hidden_size);
		} else {
			final_layernorm_bias = nullptr;
		}
		allocateArray(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	} else {
		final_layernorm_weight = nullptr;
		final_layernorm_bias = nullptr;
		output_proj_weight = nullptr;
	}

	sync_check_cuda_error_force();


	// auto tmp_qkv_w = torch::from_blob(layer_weights[0].attn_qkv_kernel, qkv_size, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
	// auto tmp_o_w = torch::from_blob(layer_weights[0].attn_out_kernel, o_size, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
	// auto tmp_ffn_1_w = torch::from_blob(layer_weights[0].ffn_fc1_weight, local_ffn_dim * hyper_param.hidden_size, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
	// auto tmp_ffn_2_w = torch::from_blob(layer_weights[0].ffn_fc2_weight, local_ffn_dim * hyper_param.hidden_size, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
	// stage_flatten_tensor_vec_ = std::vector<torch::Tensor>{tmp_qkv_w, tmp_o_w, tmp_ffn_1_w, tmp_ffn_2_w};

	#undef allocateArray
}


// allocateWeightArray - allocate memory for all weights
// This function allocates memory on GPU for all weights in the model.
template<typename T>
void GptWeight<T>::allocateWeightArray() {
	#define allocateArray(ptr, num_elem) {CUDA_CHECK(cudaMalloc(&ptr, (num_elem)*sizeof(T))); }

	const int64_t local_num_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	const int64_t local_ffn_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	// get free gpu memory
	size_t free_byte;
	size_t total_byte;
	CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

	// Allocate weights for embedding layer
	if (contain_embedding_layer) {
		allocateArray(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding) {
			allocateArray(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
		} else {
			embed_positions_weight = nullptr;
		}
	}
	else{
		embed_tokens_weight = nullptr;
		embed_positions_weight = nullptr;
	}

	auto qkv_size = hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim;
	auto o_size = local_num_q_heads*hyper_param.head_dim*hyper_param.hidden_size;
	auto ffn_size = !hyper_param.is_gated_ffn ? local_ffn_dim*hyper_param.hidden_size * 2 : local_ffn_dim*hyper_param.hidden_size * 2 + local_ffn_dim*hyper_param.hidden_size;
	fsdp_flat_param_size_ = qkv_size + o_size + ffn_size;
	// Allocate weights for each layer
	layer_weights.resize(parallelism_param.local_layer_num);
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];

		allocateArray(layer_weight.fsdp_weight, qkv_size + o_size + ffn_size);
		layer_weight.attn_qkv_kernel = layer_weight.fsdp_weight;
		// allocateArray(layer_weight.attn_qkv_kernel, hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			allocateArray(layer_weight.attn_qkv_bias, (local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		}
		// allocateArray(layer_weight.attn_out_kernel, local_num_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		layer_weight.attn_out_kernel = layer_weight.fsdp_weight + qkv_size;
		if (hyper_param.is_attn_out_biased) {
			allocateArray(layer_weight.attn_out_bias, hyper_param.hidden_size);
		}

		allocateArray(layer_weight.attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weight.attn_layernorm_bias, hyper_param.hidden_size);
		}

		layer_weight.ffn_fc1_weight = layer_weight.fsdp_weight + qkv_size + o_size;
		layer_weight.ffn_fc2_weight = layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size;

		// allocateArray(layer_weight.ffn_fc1_weight, local_ffn_dim*hyper_param.hidden_size);
		// allocateArray(layer_weight.ffn_fc2_weight, hyper_param.hidden_size*local_ffn_dim);
		if (!hyper_param.is_gated_ffn) {
			allocateArray(layer_weight.ffn_fc1_bias, local_ffn_dim);
			allocateArray(layer_weight.ffn_fc2_bias, hyper_param.hidden_size);
		} else {
			layer_weight.ffn_fc3_weight = layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size + hyper_param.hidden_size*local_ffn_dim;
			// allocateArray(layer_weight.ffn_fc3_weight, local_ffn_dim*hyper_param.hidden_size);
		}

		allocateArray(layer_weight.final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weight.final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Allocate weights for final layer
		allocateArray(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(final_layernorm_bias, hyper_param.hidden_size);
		} else {
			final_layernorm_bias = nullptr;
		}
		allocateArray(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	} else {
		final_layernorm_weight = nullptr;
		final_layernorm_bias = nullptr;
		output_proj_weight = nullptr;
	}

	sync_check_cuda_error_force();

	#undef allocateArray
}

template<typename T>
torch::Tensor GptWeight<T>::get_cos_sin_cache(){
	return torch::from_blob(this->cos_sin_cache, {hyper_param.max_position_embeddings * hyper_param.head_dim}, torch::TensorOptions().dtype(util::getTorchScalarType<T>()).device(torch::kCUDA));
}

template<typename T>
void GptWeight<T>::allocateWeightArrayFSDP() {
	#define allocateArray(ptr, num_elem) {CUDA_CHECK(cudaMalloc(&ptr, (num_elem)*sizeof(T))); }

	const int64_t local_num_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	const int64_t local_num_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	const int64_t local_ffn_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	// get free gpu memory
	size_t free_byte;
	size_t total_byte;
	CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

	// Allocate weights for embedding layer
	if (contain_embedding_layer) {
		allocateArray(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding) {
			allocateArray(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
		} else {
			embed_positions_weight = nullptr;
		}
	}
	else{
		embed_tokens_weight = nullptr;
		embed_positions_weight = nullptr;
	}

	auto qkv_size = hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim;
	auto o_size = local_num_q_heads*hyper_param.head_dim*hyper_param.hidden_size;
	auto ffn_size = !hyper_param.is_gated_ffn ? local_ffn_dim*hyper_param.hidden_size * 2 : local_ffn_dim*hyper_param.hidden_size * 2 + local_ffn_dim*hyper_param.hidden_size;
	fsdp_flat_param_size_ = qkv_size + o_size + ffn_size;
	// Allocate weights for each layer
	layer_weights.resize(parallelism_param.local_layer_num);
	allocateArray(fsdp_layer_weight.fsdp_weight, qkv_size + o_size + ffn_size);
	fsdp_layer_weight.attn_qkv_kernel = fsdp_layer_weight.fsdp_weight;
	fsdp_layer_weight.attn_out_kernel = fsdp_layer_weight.fsdp_weight + qkv_size;
	fsdp_layer_weight.ffn_fc1_weight = fsdp_layer_weight.fsdp_weight + qkv_size + o_size;
	fsdp_layer_weight.ffn_fc2_weight = fsdp_layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size;
	if (hyper_param.is_gated_ffn) {
		fsdp_layer_weight.ffn_fc3_weight = fsdp_layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size + hyper_param.hidden_size*local_ffn_dim;
	}

	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];

		// allocateArray(layer_weight.fsdp_weight, qkv_size + o_size + ffn_size);
		layer_weight.attn_qkv_kernel = fsdp_layer_weight.fsdp_weight;
		// allocateArray(layer_weight.attn_qkv_kernel, hyper_param.hidden_size*(local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			allocateArray(layer_weight.attn_qkv_bias, (local_num_q_heads+2*local_num_kv_heads)*hyper_param.head_dim);
		}
		// allocateArray(layer_weight.attn_out_kernel, local_num_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		layer_weight.attn_out_kernel = fsdp_layer_weight.fsdp_weight + qkv_size;
		if (hyper_param.is_attn_out_biased) {
			allocateArray(layer_weight.attn_out_bias, hyper_param.hidden_size);
		}

		allocateArray(layer_weight.attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weight.attn_layernorm_bias, hyper_param.hidden_size);
		}

		// allocateArray(layer_weight.ffn_fc1_weight, local_ffn_dim*hyper_param.hidden_size);
		// allocateArray(layer_weight.ffn_fc2_weight, hyper_param.hidden_size*local_ffn_dim);
		layer_weight.ffn_fc1_weight = fsdp_layer_weight.fsdp_weight + qkv_size + o_size;
		layer_weight.ffn_fc2_weight = fsdp_layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size;
		if (!hyper_param.is_gated_ffn) {
			allocateArray(layer_weight.ffn_fc1_bias, local_ffn_dim);
			allocateArray(layer_weight.ffn_fc2_bias, hyper_param.hidden_size);
			allocateArray(layer_weight.ffn_fc1_weight, local_ffn_dim*hyper_param.hidden_size);
		} else {
			layer_weight.ffn_fc3_weight = layer_weight.fsdp_weight + qkv_size + o_size + local_ffn_dim*hyper_param.hidden_size + hyper_param.hidden_size*local_ffn_dim;
			// allocateArray(layer_weight.ffn_fc3_weight, local_ffn_dim*hyper_param.hidden_size);
		}

		allocateArray(layer_weight.final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(layer_weight.final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Allocate weights for final layer
		allocateArray(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			allocateArray(final_layernorm_bias, hyper_param.hidden_size);
		} else {
			final_layernorm_bias = nullptr;
		}
		allocateArray(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	} else {
		final_layernorm_weight = nullptr;
		final_layernorm_bias = nullptr;
		output_proj_weight = nullptr;
	}

	sync_check_cuda_error_force();

	#undef allocateArray
}


// freeWeightArray - free memory for all weights
// This function frees memory on GPU for all weights in the model.
template<typename T>
void GptWeight<T>::freeWeightArray() {
	#define freeArray(ptr) {if ((ptr) != nullptr) { CUDA_CHECK(cudaFree(ptr));} }

	// Free weights for embedding layer
	if (contain_embedding_layer) {
		freeArray(embed_tokens_weight);
		freeArray(embed_positions_weight);
	}

	// Free weights for each layer
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];

		freeArray(layer_weight.attn_qkv_kernel);
		freeArray(layer_weight.attn_qkv_bias);
		freeArray(layer_weight.attn_out_kernel);
		freeArray(layer_weight.attn_out_bias);

		freeArray(layer_weight.attn_layernorm_weight);
		freeArray(layer_weight.attn_layernorm_bias);

		freeArray(layer_weight.ffn_fc1_weight);
		freeArray(layer_weight.ffn_fc1_bias);
		freeArray(layer_weight.ffn_fc2_weight);
		freeArray(layer_weight.ffn_fc2_bias);
		freeArray(layer_weight.ffn_fc3_weight);
		freeArray(layer_weight.ffn_fc13_weight);

		freeArray(layer_weight.final_layernorm_weight);
		freeArray(layer_weight.final_layernorm_bias);
	}

	if (parallelism_param.is_last_stage()){
		// Free weights for final layer
		freeArray(final_layernorm_weight);
		freeArray(final_layernorm_bias);
		freeArray(output_proj_weight);
	}

	sync_check_cuda_error_force();

	#undef freeArray
}

template<typename T>
void GptWeight<T>::loadTensor_qkv_weight_kernel_or_bias(const uint32_t merge_dim, T* to_ptr, const std::string model_dir, const std::string key, const int64_t expect_size) {
	// Please set merge_dim to 1 when loading qkv_weight_kernel and 0 when loading qkv_weight_bias
	const std::string basename = model_dir + key + ".tp_";
	const uint32_t num_tensor = 8 / parallelism_param.tensor_para_size;
	const uint32_t begin_num = parallelism_param.tensor_para_rank * num_tensor;
	const uint32_t end_num = (parallelism_param.tensor_para_rank + 1) * num_tensor;
	
	const int64_t single_tensor_size = expect_size / 8;
	const int64_t single_worker_size = single_tensor_size * num_tensor;

	std::vector<torch::Tensor> qs(num_tensor), ks(num_tensor), vs(num_tensor);
	for (uint32_t i = begin_num; i < end_num; ++i) {
		// Deal with cases where num_q_heads or num_kv_heads is not divisible by 8, like in OPT-125M
		int64_t q_heads_in_this_tensor = hyper_param.num_q_heads / 8 + 
			(i != end_num-1 ? 0 : hyper_param.num_q_heads%8);
		int64_t kv_heads_in_this_tensor = hyper_param.num_kv_heads / 8 +
			(i != end_num-1 ? 0 : hyper_param.num_kv_heads%8);
		int64_t heads_in_this_tensor = q_heads_in_this_tensor + 2*kv_heads_in_this_tensor;
		torch::Tensor cur_tensor = torch::jit::load(basename + std::to_string(i) + ".pt", torch::kCPU).attr(key).toTensor();
		cur_tensor = merge_dim == 1 ?
			cur_tensor.view({hyper_param.hidden_size, heads_in_this_tensor, hyper_param.head_dim}) : 
			cur_tensor.view({heads_in_this_tensor, hyper_param.head_dim});
		// cur_tensor: [hidden_size, (num_q_heads+2*num_kv_heads)//8, head_dim] (when loading qkv_weight_kernel)
		// or [(num_q_heads+2*num_kv_heads)//8, head_dim] (when loading qkv_weight_bias)
		qs[i-begin_num] = cur_tensor.slice(merge_dim, 0, q_heads_in_this_tensor);
		ks[i-begin_num] = cur_tensor.slice(merge_dim, q_heads_in_this_tensor, q_heads_in_this_tensor+kv_heads_in_this_tensor);
		vs[i-begin_num] = cur_tensor.slice(merge_dim, q_heads_in_this_tensor+kv_heads_in_this_tensor, q_heads_in_this_tensor+2*kv_heads_in_this_tensor);
	}

	torch::Tensor final_q = torch::cat(qs, merge_dim);	// [hidden_size, num_q_heads*head_dim//tensor_para_size] or [num_q_heads*head_dim//tensor_para_size]
	torch::Tensor final_k = torch::cat(ks, merge_dim);	// [hidden_size, num_kv_heads*head_dim//tensor_para_size] or [num_kv_heads*head_dim//tensor_para_size]
	torch::Tensor final_v = torch::cat(vs, merge_dim);
	torch::Tensor final_tensor = torch::cat({final_q, final_k, final_v}, merge_dim);
	if (final_tensor.numel() != single_worker_size) {
		std::cerr << "Gpt<T>::load() - " << basename << " size not match" << std::endl;
		throw std::runtime_error("");
	}
	if (final_tensor.scalar_type() != util::getTorchScalarType<T>()) {
		std::cerr << "Gpt<T>::load() - " << basename << " type not match" << std::endl;
		throw std::runtime_error("");
	}
	CUDA_CHECK(cudaMemcpy(to_ptr, (T*)final_tensor.data_ptr(), single_worker_size*sizeof(T), cudaMemcpyHostToDevice));
}

// Load tensor parallel weight
template<typename T>
void GptWeight<T>::loadTensor_tp(const uint32_t dim, T* to_ptr, const std::string model_dir, const std::string key,  const int64_t expect_size, int64_t start_loc) {
	// example: decoder.layers.0.self_attn.qkv_proj.weight.tp_0.pt
	const std::string basename = model_dir + key + ".tp_";
	const uint32_t num_tensor = 8 / parallelism_param.tensor_para_size;
	const uint32_t begin_num = parallelism_param.tensor_para_rank * num_tensor;
	const uint32_t end_num = (parallelism_param.tensor_para_rank + 1) * num_tensor;
	
	const int64_t single_tensor_size = expect_size / 8;
	const int64_t single_worker_size = single_tensor_size * num_tensor;

	// Load num_tensor tensors by torch::jit::load
	std::vector<torch::Tensor> tensors(num_tensor);
	for (uint32_t i = begin_num; i < end_num; i++){
		std::cerr << "Gpt<T>::loading - " << basename + std::to_string(i) + ".pt"  << std::endl;
		tensors[i-begin_num] = torch::jit::load(basename + std::to_string(i) + ".pt", torch::kCPU).attr(key).toTensor();
	}

	// concat tensors to one tensor
	torch::Tensor tmp_tensor = torch::cat(tensors, dim).clone();
	if (tmp_tensor.numel() != single_worker_size){
		std::cerr << "Gpt<T>::load() - " << basename << " size not match" << std::endl;
		throw std::runtime_error("");
	}
	if (tmp_tensor.scalar_type() != util::getTorchScalarType<T>()){
		std::cerr << "Gpt<T>::load() - " << basename << " type not match" << std::endl;
		throw std::runtime_error("");
	}
	CUDA_CHECK(cudaMemcpy(to_ptr + start_loc,  (T*)tmp_tensor.data_ptr(), single_worker_size*sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void GptWeight<T>::loadTensor_all(T* to_ptr, const std::string model_dir, const std::string key, const int64_t expect_size) {
	auto tensor_path = model_dir + key + ".pt";
	if (!std::filesystem::exists(tensor_path)) {
		std::cerr << "Gpt<T>::load() - " << tensor_path << " not found" << std::endl;
		throw std::runtime_error("");
	}
	torch::jit::Module tmp = torch::jit::load(tensor_path, torch::kCPU);
	torch::Tensor tmp_tensor = tmp.attr(key).toTensor();
	if (tmp_tensor.numel() != expect_size){
		std::cerr << "Gpt<T>::load() - " << tensor_path << " size not match" << std::endl;
		throw std::runtime_error("");
	}
	if (tmp_tensor.scalar_type() != util::getTorchScalarType<T>()){
		std::cerr << "Gpt<T>::load() - " << tensor_path << " type not match" << std::endl;
		throw std::runtime_error("");
	}
	CUDA_CHECK(cudaMemcpy(to_ptr,  (T*) tmp_tensor.data_ptr(), expect_size*sizeof(T), cudaMemcpyHostToDevice));
}

// GptWeight<T>::loadWeight() - Load weights from (converted) pt files
// This function loads weights from a series of pt file. The pt files are converted from
// the original pytorch model using the script `scripts/convert-XXX.py`.
template<typename T>
void GptWeight<T>::loadWeight(const std::string& model_path) {
	// Make sure `model_path` exists
	if (!std::filesystem::exists(model_path)) {
		std::cerr << "Gpt<T>::load() - " << model_path << " not found" << std::endl;
		throw std::runtime_error("");
	}

	// We only support tensor parallel size = 1 when the heads cannot be divided evenly
	if ((hyper_param.num_q_heads % parallelism_param.tensor_para_size != 0 || hyper_param.num_kv_heads % parallelism_param.tensor_para_size != 0)
		&& parallelism_param.tensor_para_size != 1) {
		std::cerr << "Gpt<T>::load() - tensor parallel size must be 1 when the heads cannot be divided evenly by 8" << std::endl;
		throw std::runtime_error("");
	}
	
	if (contain_embedding_layer){
		// Load weights for embedding layer
		loadTensor_all(embed_tokens_weight, model_path, "decoder.embed_tokens.weight", hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding)
			loadTensor_all(embed_positions_weight, model_path, "decoder.embed_positions.weight", hyper_param.max_position_embeddings*hyper_param.hidden_size);
	}

	// concat fc1 & fc3 tensor
	// T* ffn_fc1_tmp_weight = nullptr;
	// T* ffn_fc3_tmp_weight = nullptr;
	// allocateArray(ffn_fc1_tmp_weight, hyper_param.local_ffn_dim*hyper_param.hidden_size);
	// allocateArray(ffn_fc3_tmp_weight, hyper_param.local_ffn_dim*hyper_param.hidden_size);
	// auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

	// Load weights for each layer
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];
		int layer_id = parallelism_param.layer_begin + i;
		
		loadTensor_qkv_weight_kernel_or_bias(1, layer_weight.attn_qkv_kernel, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn.qkv_proj.weight", hyper_param.hidden_size*(hyper_param.num_q_heads+2*hyper_param.num_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			loadTensor_qkv_weight_kernel_or_bias(0, layer_weight.attn_qkv_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn.qkv_proj.bias", (hyper_param.num_q_heads+2*hyper_param.num_kv_heads)*hyper_param.head_dim);
		}
		loadTensor_tp(0, layer_weight.attn_out_kernel, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn.out_proj.weight", hyper_param.num_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		if (hyper_param.is_attn_out_biased) {
			loadTensor_all(layer_weight.attn_out_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn.out_proj.bias", hyper_param.hidden_size);
		}

		loadTensor_all(layer_weight.attn_layernorm_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn_layer_norm.weight", hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			loadTensor_all(layer_weight.attn_layernorm_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".self_attn_layer_norm.bias", hyper_param.hidden_size);
		}

		// loadTensor_tp(0, layer_weight.ffn_fc1_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc1.weight", hyper_param.ffn_inter_dim*hyper_param.hidden_size);
		loadTensor_tp(1, layer_weight.ffn_fc2_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc2.weight", hyper_param.hidden_size*hyper_param.ffn_inter_dim);
		
		if (!hyper_param.is_gated_ffn) {
			loadTensor_tp(0, layer_weight.ffn_fc1_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".fc1.bias", hyper_param.ffn_inter_dim);
			loadTensor_all(layer_weight.ffn_fc2_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".fc2.bias", hyper_param.hidden_size);
			loadTensor_tp(0, layer_weight.ffn_fc1_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc1.weight", hyper_param.ffn_inter_dim*hyper_param.hidden_size);
		} else {
			loadTensor_tp(0, layer_weight.ffn_fc13_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc1.weight", hyper_param.ffn_inter_dim*hyper_param.hidden_size);
			// loadTensor_tp(0, layer_weight.ffn_fc3_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc3.weight", hyper_param.ffn_inter_dim*hyper_param.hidden_size);
			loadTensor_tp(0, layer_weight.ffn_fc13_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".fc3.weight", hyper_param.ffn_inter_dim*hyper_param.hidden_size, hyper_param.ffn_inter_dim*hyper_param.hidden_size);

			// at::Tensor ffn_fc1_tmp_weight_tsr = torch::from_blob(const_cast<T*>(ffn_fc1_tmp_weight),
            //                        {hyper_param.hidden_size,hyper_param.ffn_inter_dim}, options);
			// at::Tensor ffn_fc3_tmp_weight_tsr = torch::from_blob(const_cast<T*>(ffn_fc3_tmp_weight),
            //                        {hyper_param.hidden_size,hyper_param.ffn_inter_dim}, options);
			// std::vector<torch::Tensor> tensors(2);
			// tensors[0] = ffn_fc1_tmp_weight_tsr;
			// tensors[1] = ffn_fc3_tmp_weight_tsr;
			// at::Tensor ffn_fc13_tmp_weight_tsr = torch::cat(tensors, 0).clone();
			// CUDA_CHECK(cudaMemcpy(layer_weight.ffn_fc13_weight,  (T*) ffn_fc13_tmp_weight_tsr.data_ptr(), 2*hyper_param.hidden_size*hyper_param.ffn_inter_dim*sizeof(T), cudaMemcpyDeviceToDevice));
		}

		loadTensor_all(layer_weight.final_layernorm_weight, model_path, "decoder.layers."+std::to_string(layer_id)+".final_layer_norm.weight", hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			loadTensor_all(layer_weight.final_layernorm_bias, model_path, "decoder.layers."+std::to_string(layer_id)+".final_layer_norm.bias", hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()){
		// Load weights for final layer
		loadTensor_all(final_layernorm_weight, model_path, "decoder.layer_norm.weight", hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			loadTensor_all(final_layernorm_bias, model_path, "decoder.layer_norm.bias", hyper_param.hidden_size);
		}

		loadTensor_all(output_proj_weight, model_path, "decoder.output_projection.weight", hyper_param.vocab_size*hyper_param.hidden_size);
	}

	// Check if there is any error
	sync_check_cuda_error_force();
}


template<typename T>
void GptWeight<T>::initDummyWeight() {
	// We only support tensor parallel size = 1 when the heads cannot be divided evenly
	if ((hyper_param.num_q_heads % parallelism_param.tensor_para_size != 0 || hyper_param.num_kv_heads % parallelism_param.tensor_para_size != 0)
		&& parallelism_param.tensor_para_size != 1) {
		std::cerr << "Gpt<T>::load() - tensor parallel size must be 1 when the heads cannot be divided evenly by 8" << std::endl;
		throw std::runtime_error("");
	}

	// Use a fixed seed here, so that the weights are the same across different runs,
	// which makes it possible to check whether the results are correct even under
	// dummy weights
	torch::manual_seed(0);

	// initDummyTensor - Fill an array starting from `addr` with `numel` elements with uniform random numbers in [-1e-3, 1e-3]
	// We create a torch::Tensor which is also pointing to the same memory address, and use torch::Tensor::uniform_() to fill it
	auto initDummyTensor = [](T* addr, int64_t numel) {
		// torch::from_blob: Exposes the given data as a Tensor without taking ownership of the original data.
		// So when tmp is destructed, the memory won't be freed
		torch::Tensor tmp = torch::from_blob(addr, {numel}, torch::TensorOptions().dtype(util::getTorchScalarType<T>()).device(torch::kCUDA));
		tmp.uniform_(-1e-3, 1e-3);
	};
	
	// Load weights for embedding layer
	if (contain_embedding_layer) {
		initDummyTensor(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding)
			initDummyTensor(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
	}
	
	int64_t num_local_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	int64_t num_local_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	int64_t local_ffn_inter_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	// Load weights for each layer
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];
		
		initDummyTensor(layer_weight.attn_qkv_kernel, hyper_param.hidden_size*(num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			initDummyTensor(layer_weight.attn_qkv_bias, (num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		}
		initDummyTensor(layer_weight.attn_out_kernel, num_local_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		if (hyper_param.is_attn_out_biased) {
			initDummyTensor(layer_weight.attn_out_bias, hyper_param.hidden_size);
		}

		initDummyTensor(layer_weight.attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.attn_layernorm_bias, hyper_param.hidden_size);
		}

		// initDummyTensor(layer_weight.ffn_fc1_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		initDummyTensor(layer_weight.ffn_fc2_weight, hyper_param.hidden_size*local_ffn_inter_dim);

		if (!hyper_param.is_gated_ffn) {
			initDummyTensor(layer_weight.ffn_fc1_bias, local_ffn_inter_dim);
			initDummyTensor(layer_weight.ffn_fc2_bias, hyper_param.hidden_size);
			initDummyTensor(layer_weight.ffn_fc1_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		} else {
			initDummyTensor(layer_weight.ffn_fc13_weight, 2*local_ffn_inter_dim*hyper_param.hidden_size);
		}

		initDummyTensor(layer_weight.final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Load weights for final layer
		initDummyTensor(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(final_layernorm_bias, hyper_param.hidden_size);
		}
		initDummyTensor(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	}

	// Check if there is any error
	sync_check_cuda_error_force();
	printf("Dummy weight inited!\n");
}

template<typename T>
std::vector<torch::Tensor>  GptWeight<T>::GetFlattenWeight() {
	//  torch::from_blob(stage_flatten_weight_, stage_flatten_weight_size_, torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA));
	return stage_flatten_tensor_vec_;
}


template<typename T>
void GptWeight<T>::initDummyWeightFSDPContext() {
	// We only support tensor parallel size = 1 when the heads cannot be divided evenly
	if ((hyper_param.num_q_heads % parallelism_param.tensor_para_size != 0 || hyper_param.num_kv_heads % parallelism_param.tensor_para_size != 0)
		&& parallelism_param.tensor_para_size != 1) {
		std::cerr << "Gpt<T>::load() - tensor parallel size must be 1 when the heads cannot be divided evenly by 8" << std::endl;
		throw std::runtime_error("");
	}

	// Use a fixed seed here, so that the weights are the same across different runs,
	// which makes it possible to check whether the results are correct even under
	// dummy weights
	torch::manual_seed(0);

	// initDummyTensor - Fill an array starting from `addr` with `numel` elements with uniform random numbers in [-1e-3, 1e-3]
	// We create a torch::Tensor which is also pointing to the same memory address, and use torch::Tensor::uniform_() to fill it
	auto initDummyTensor = [](T* addr, int64_t numel) {
		// torch::from_blob: Exposes the given data as a Tensor without taking ownership of the original data.
		// So when tmp is destructed, the memory won't be freed
		torch::Tensor tmp = torch::from_blob(addr, {numel}, torch::TensorOptions().dtype(util::getTorchScalarType<T>()).device(torch::kCUDA));
		tmp.uniform_(-1e-3, 1e-3);
		return tmp;
	};
	
	// Load weights for embedding layer
	if (contain_embedding_layer) {
		initDummyTensor(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding)
			initDummyTensor(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
	}
	
	int64_t num_local_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	int64_t num_local_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	int64_t local_ffn_inter_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	auto qkv_size = hyper_param.hidden_size*(num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim;
	auto o_size = num_local_q_heads*hyper_param.head_dim*hyper_param.hidden_size;
	auto ffn_size = !hyper_param.is_gated_ffn ? local_ffn_inter_dim*hyper_param.hidden_size * 2 : local_ffn_inter_dim*hyper_param.hidden_size * 2 + local_ffn_inter_dim*hyper_param.hidden_size;

	// Load weights for each layer
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];
		
		torch::Tensor flat_param = initDummyTensor(layer_weight.fsdp_weight, qkv_size + o_size + ffn_size);
		// auto fsdp_param_vec = flat_param.split({qkv_size, o_size, ffn_size});
		// layer_weight.attn_qkv_kernel = fsdp_param_vec[0].data_ptr<torch::TensorOptions().dtype(util::getTorchScalarType<T>())>();
		// layer_weight.attn_out_bias = fsdp_param_vec[1].data_ptr<torch::TensorOptions().dtype(util::getTorchScalarType<T>())>();
		// layer_weight.ffn_fc3_weight = fsdp_param_vec[2].data_ptr<torch::TensorOptions().dtype(util::getTorchScalarType<T>())>();

		// auto tmp_ptr = fsdp_param_vec[0].data_ptr();
		// layer_weight.attn_qkv_kernel = static_cast<T*>(fsdp_param_vec[0].data_ptr());
		// layer_weight.attn_out_bias = static_cast<T*>(fsdp_param_vec[0].data_ptr());
		// layer_weight.ffn_fc3_weight = static_cast<T*>(fsdp_param_vec[0].data_ptr());
		// std::cout << "flat_param address: " << flat_param << std::endl;
		// layer_weight.attn_qkv_kernel = static_cast<T*>(flat_param.data_ptr());
		// std::cout << "attn_qkv_kernel address: " << layer_weight.attn_qkv_kernel << std::endl;
		// layer_weight.attn_out_bias = static_cast<T*>(flat_param.data_ptr());
		// layer_weight.ffn_fc3_weight = static_cast<T*>(flat_param.data_ptr());


		// initDummyTensor(layer_weight.attn_qkv_kernel, hyper_param.hidden_size*(num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			initDummyTensor(layer_weight.attn_qkv_bias, (num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		}
		// initDummyTensor(layer_weight.attn_out_kernel, num_local_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		if (hyper_param.is_attn_out_biased) {
			initDummyTensor(layer_weight.attn_out_bias, hyper_param.hidden_size);
		}

		initDummyTensor(layer_weight.attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.attn_layernorm_bias, hyper_param.hidden_size);
		}

		// initDummyTensor(layer_weight.ffn_fc1_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		// initDummyTensor(layer_weight.ffn_fc2_weight, hyper_param.hidden_size*local_ffn_inter_dim);

		if (!hyper_param.is_gated_ffn) {
			initDummyTensor(layer_weight.ffn_fc1_bias, local_ffn_inter_dim);
			initDummyTensor(layer_weight.ffn_fc2_bias, hyper_param.hidden_size);
		} 
		// else {
		// 	initDummyTensor(layer_weight.ffn_fc3_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		// }

		initDummyTensor(layer_weight.final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Load weights for final layer
		initDummyTensor(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(final_layernorm_bias, hyper_param.hidden_size);
		}
		initDummyTensor(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	}

	// Check if there is any error
	sync_check_cuda_error_force();
	printf("Dummy weight inited!\n");
}


template<typename T>
void GptWeight<T>::initDummyWeightFSDP() {
	// We only support tensor parallel size = 1 when the heads cannot be divided evenly
	if ((hyper_param.num_q_heads % parallelism_param.tensor_para_size != 0 || hyper_param.num_kv_heads % parallelism_param.tensor_para_size != 0)
		&& parallelism_param.tensor_para_size != 1) {
		std::cerr << "Gpt<T>::load() - tensor parallel size must be 1 when the heads cannot be divided evenly by 8" << std::endl;
		throw std::runtime_error("");
	}

	// Use a fixed seed here, so that the weights are the same across different runs,
	// which makes it possible to check whether the results are correct even under
	// dummy weights
	torch::manual_seed(0);

	// initDummyTensor - Fill an array starting from `addr` with `numel` elements with uniform random numbers in [-1e-3, 1e-3]
	// We create a torch::Tensor which is also pointing to the same memory address, and use torch::Tensor::uniform_() to fill it
	auto initDummyTensor = [](T* addr, int64_t numel) {
		// torch::from_blob: Exposes the given data as a Tensor without taking ownership of the original data.
		// So when tmp is destructed, the memory won't be freed
		torch::Tensor tmp = torch::from_blob(addr, {numel}, torch::TensorOptions().dtype(util::getTorchScalarType<T>()).device(torch::kCUDA));
		tmp.uniform_(-1e-3, 1e-3);
		return tmp;
	};
	
	// Load weights for embedding layer
	if (contain_embedding_layer) {
		initDummyTensor(embed_tokens_weight, hyper_param.vocab_size*hyper_param.hidden_size);
		if (!hyper_param.is_rotary_posi_embedding)
			initDummyTensor(embed_positions_weight, hyper_param.max_position_embeddings*hyper_param.hidden_size);
	}
	
	int64_t num_local_q_heads = hyper_param.num_q_heads / parallelism_param.tensor_para_size;
	int64_t num_local_kv_heads = hyper_param.num_kv_heads / parallelism_param.tensor_para_size;
	int64_t local_ffn_inter_dim = hyper_param.ffn_inter_dim / parallelism_param.tensor_para_size;

	// FSDP. Only discard QKVO and FFN proj weight
	auto qkv_size = hyper_param.hidden_size*(num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim;
	auto o_size = num_local_q_heads*hyper_param.head_dim*hyper_param.hidden_size;
	auto ffn_size = !hyper_param.is_gated_ffn ? local_ffn_inter_dim*hyper_param.hidden_size * 2 : local_ffn_inter_dim*hyper_param.hidden_size * 2 + local_ffn_inter_dim*hyper_param.hidden_size;

	initDummyTensor(fsdp_layer_weight.fsdp_weight, qkv_size + o_size + ffn_size);
	// auto fsdp_param_vec = flat_param.split({qkv_size, o_size, ffn_size});
	// fsdp_layer_weight.attn_qkv_kernel = static_cast<T*>(fsdp_param_vec[0].data_ptr());
	// fsdp_layer_weight.attn_out_bias = static_cast<T*>(fsdp_param_vec[1].data_ptr());
	// fsdp_layer_weight.ffn_fc3_weight = static_cast<T*>(fsdp_param_vec[2].data_ptr());
	// printf("init fsdp data succ");

	// Load weights for each layer
	for (int64_t i = 0; i < parallelism_param.local_layer_num; ++i) {
		GptLayerWeight<T>& layer_weight = layer_weights[i];
		
		// initDummyTensor(layer_weight.attn_qkv_kernel, hyper_param.hidden_size*(num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		if (hyper_param.is_attn_qkv_biased) {
			initDummyTensor(layer_weight.attn_qkv_bias, (num_local_q_heads+2*num_local_kv_heads)*hyper_param.head_dim);
		}
		// initDummyTensor(layer_weight.attn_out_kernel, num_local_q_heads*hyper_param.head_dim*hyper_param.hidden_size);
		if (hyper_param.is_attn_out_biased) {
			initDummyTensor(layer_weight.attn_out_bias, hyper_param.hidden_size);
		}

		initDummyTensor(layer_weight.attn_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.attn_layernorm_bias, hyper_param.hidden_size);
		}

		// initDummyTensor(layer_weight.ffn_fc1_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		// initDummyTensor(layer_weight.ffn_fc2_weight, hyper_param.hidden_size*local_ffn_inter_dim);

		if (!hyper_param.is_gated_ffn) {
			initDummyTensor(layer_weight.ffn_fc1_bias, local_ffn_inter_dim);
			initDummyTensor(layer_weight.ffn_fc2_bias, hyper_param.hidden_size);
		} 
		// else {
		// 	initDummyTensor(layer_weight.ffn_fc3_weight, local_ffn_inter_dim*hyper_param.hidden_size);
		// }

		initDummyTensor(layer_weight.final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(layer_weight.final_layernorm_bias, hyper_param.hidden_size);
		}
	}

	if (parallelism_param.is_last_stage()) {
		// Load weights for final layer
		initDummyTensor(final_layernorm_weight, hyper_param.hidden_size);
		if (!hyper_param.is_rmsnorm) {
			initDummyTensor(final_layernorm_bias, hyper_param.hidden_size);
		}
		initDummyTensor(output_proj_weight, hyper_param.vocab_size*hyper_param.hidden_size);
	}

	// Check if there is any error
	sync_check_cuda_error_force();
	printf("Dummy weight inited!\n");
}

template class GptWeight<half>;
template class GptWeight<float>;

}	// namespace st::model
