#pragma once

#include <string>

#include <torch/script.h>

#include "model/gpt/gpt_base.h"
#include "model/gpt/gpt.h"

namespace st::model {

// Please refer to gpt_base.h for the design of GPTBase, Gpt, and GptOpBase.
class GptOpBase : public torch::CustomClassHolder {
public:
	GptBase* gpt;	// A pointer to GptBase, which can be Gpt<T> for any T.
	bool weight_loaded;

	GptOpBase(
		std::string inference_dtype,
		GptHyperParam hyper_param,
		GptPagedAttnParam pagedattn_param,
		GptParallelismParam parallelism_param
	);

	~GptOpBase();

	void loadWeight(const std::string& weight_path);
	void initDummyWeight();
	void initDummyWeightFSDP();
	std::vector<torch::Tensor>  GetFlattenWeight();
	void LazyInitWeight(bool enable_ipc_mem, bool is_alloc);

    std::vector<int64_t> forward(
        const std::vector<std::vector<int64_t>> &input_tokens_batched,
	    const std::vector<int64_t> &first_token_indexes,
        torch::Tensor &k_cache,
        torch::Tensor &v_cache,
        const std::vector<std::vector<int64_t>> &block_table
    );

	std::vector<int64_t> pipelinedforward(
        const std::vector<std::vector<int64_t>> &input_tokens_batched,
	    const std::vector<int64_t> &first_token_indexes,
        torch::Tensor &k_cache,
        torch::Tensor &v_cache,
        const std::vector<std::vector<int64_t>> &block_table
    );

	torch::Tensor GetCosSinCache();

	void prologue(
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes, // [batchsize]
		torch::Tensor &k_cache, // [num_blocks, num_heads, block_size, head_dim]
		torch::Tensor &v_cache, // [num_blocks, num_heads, block_size, head_dim]
		const std::vector<std::vector<int64_t>> &block_table);

	void ExecuteDecoderLayer(int64_t layer_id);

	std::vector<int64_t> epilogue();

	void init_communicator(const std::vector<int64_t> tp_id, const std::vector<int64_t> pp_id);

	void init_p2p_communicator(const std::vector<int64_t> peer_ids);

	void send_weight(int64_t layer_id);

	void recv_weight(int64_t layer_id);

	void wait_stream();

	void nccl_group_start();

	void nccl_group_end();

private:
	int64_t *d_block_table_ = nullptr;
	bool has_prologue_run_ = false;
	bool early_return_ = false;

};

}