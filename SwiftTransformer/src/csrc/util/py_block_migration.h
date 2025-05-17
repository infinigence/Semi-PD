#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <torch/extension.h>
#include "cuda_utils.h"

namespace st::util {

std::vector<int64_t> get_ipc_mem_handle(torch::Tensor tensor);

bool register_ipc_mem_handle(
	std::vector<int64_t> k_cache_handle_vec,
	std::vector<int64_t> v_cache_handle_vec,
	int64_t num_layers,
	int64_t num_heads,
	const std::vector<int64_t> &context_parallel_config,
	const std::vector<int64_t> &decoding_parallel_config
);

void migrate_blocks(
	const int64_t context_pp_size,
	const int64_t context_tp_size,

	const std::vector<int64_t> &context_block_indexes,

	const int64_t decoding_pp_size,
	const int64_t decoding_tp_size,

	const int64_t decoding_pp_rank,
	const int64_t decoding_tp_rank,

	const std::vector<int64_t> &decoding_block_indexes,

	torch::Tensor decoding_worker_k_cache,
	torch::Tensor decoding_worker_v_cache
);

std::vector<torch::Tensor> share_kv_cache_memory(
	const std::vector<int64_t> &context_parallel_config,	// Generated via ParallelConfig.to_list()
	const std::vector<int64_t> &decoding_parallel_config,
	int64_t tensor_size
	);

std::vector<torch::Tensor> share_weight_memory(
	const std::vector<int64_t> &context_parallel_config,	// Generated via ParallelConfig.to_list()
	int64_t tensor_size
	);

bool register_ipc_test_addr(
	std::vector<int64_t> handle_vec
);

torch::Tensor share_test_ipc_tensor(int64_t tensor_size);


bool register_weight_ipc_mem_handle(
	std::vector<std::vector<int64_t>> weight_handle_vec,
	const std::vector<int64_t> &context_parallel_config);


// template <typename T>
void* get_ipc_weight_ptr(int64_t index);
// void* get_ipc_weight_ptr(int64_t pp_rank, int64_t tp_rank, half** out_ptr);
// void* get_ipc_weight_ptr(int64_t pp_rank, int64_t tp_rank, float** out_ptr);


int64_t get_device_available_SMs();

} // namespace st::util
