#pragma once

#include <string>

#include "util/cublas_wrapper.h"
#include "util/cuda_utils.h"
#include "util/nccl_utils.h"
#include "util/torch_utils.h"

#include "gpt_weight.h"
#include "gpt_base.h"

#include <nccl.h>

namespace st::model {

// Please refer to gpt_base.h for the design of GptBase, Gpt, GptOpBase, and XXXop.
template<typename T>
class Gpt : public GptBase {
private:
	GptWeight<T> weight;
	util::CublasWrapper cublas_wrapper;
	util::NcclComm tensor_para_comm, pipeline_para_comm;
	util::NcclComm p2p_comm;

	// Buffers for inputs & input metadata
	RemallocableArray<T> d_decoder_input;	// [num_tokens, hidden_size]
	RemallocableArray<T> d_decoder_output;	// [num_tokens, hidden_size]
	RemallocableArray<int32_t> d_input_lens;	// [batch_size]
	RemallocableArray<int64_t> d_sum_prev_input_lens;	// [batch_size]

	// Buffers for input embedding
	RemallocableArray<int64_t> d_token_ids;		// [num_tokens]
	RemallocableArray<int64_t> d_position_ids;	// [num_tokens]

	// Buffers for input indexing
	RemallocableArray<int64_t> ith_context_req_req_index;	// [batch_size]
	RemallocableArray<int32_t> ith_context_req_token_index;	// [batch_size]
	RemallocableArray<int64_t> ith_decoding_req_req_index;	// [batch_size]
	RemallocableArray<int64_t> ith_decoding_req_token_index;// [batch_size]

	// Buffers for each layer's internal computation
	RemallocableArray<T> qkv_buf;		// [num_tokens+15, local_q_head_num + 2*local_kv_head_num, head_dim]. Please refer to fused_context_stage_attention.cu for the reason of +15 here
	RemallocableArray<T> attn_out_buf;	// [num_tokens, local_q_head_num, head_dim]
	RemallocableArray<T> ffn_inter_buf_1;	// [num_tokens, local_ffn_inter_dim]
	RemallocableArray<T> ffn_inter_buf_2;	// [num_tokens, local_ffn_inter_dim], only used when is_gated_ffn = true
	RemallocableArray<float> context_stage_kernel_m_buf;	// [local_q_head_num, num_tokens]
	RemallocableArray<float> context_stage_kernel_l_buf;	// [local_q_head_num, num_tokens]

	// Buffers for forwardDecoder
	RemallocableArray<T> attention_out;	// [num_tokens, hidden_size]

	// Buffers for output projection
	RemallocableArray<T> output_projection_last_tokens_buf;	// [batch_size, hidden_dim]
	RemallocableArray<T> output_projection_buf;	// [batch_size, vocab_size]
	RemallocableArray<int64_t> output_projection_result_buf;	// [batch_size]

	int64_t max_context_req_len = 0, max_decoding_req_len = 0;
	int64_t num_tokens = 0;
	int64_t num_context_reqs = 0, num_decoding_reqs = 0;

	int64_t* h_input_lens = nullptr;
    bool* h_is_context_stage = nullptr;
	int64_t* h_sum_prev_input_lens = nullptr;
	void * d_k_cache = nullptr;
    void * d_v_cache = nullptr;
	int64_t batch_size = 0;
	int32_t* d_block_table_ = nullptr;
	cudaEvent_t recv_event_;
	cudaStream_t compute_stream_ = nullptr;

public:
	Gpt(const GptHyperParam& hyper_param, 
		const GptPagedAttnParam& pagedattn_param, 
		GptParallelismParam& parallelism_param = GptParallelismParam()
	);
	~Gpt() override;

	void setPagedattnParam(const GptPagedAttnParam& pagedattn_param);
	void setParallelismParam(const GptParallelismParam& parallelism_param);

	// Init communicator for NCCL.
	// Args:
	//	tp_id: NCCL unique ID for tensor parallelism.
	//	pp_id: NCCL unique ID for pipeline parallelism.
	void init_communicator(const ncclUniqueId& tp_id, const ncclUniqueId& pp_id);

	void init_p2p_communicator(const ncclUniqueId& p2p_id);

	void send_weight(int64_t layer_id);

	void recv_weight(int64_t layer_id);

	void wait_stream();

	void nccl_group_start();

	torch::Tensor GetCosSinCache();

	void nccl_group_end();
	void getInputPosiIds(
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,
		const int64_t num_tokens
	);

	void inputBatchEmbedAndPosiEncode(
		T* d_output,
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const int64_t num_tokens
	);

	void inputBatchEmbedAndPosiEncode(
		T* d_output,
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,
		const int64_t num_tokens
	);

	void selectOutputTokenBatched(
		int64_t* h_result_token,
		const T* d_input,
		int64_t num_tokens,
		const int64_t* first_token_indexes,
		int64_t batch_size
	);
	
	void forwardDecoder(
		T* d_output,
		const T* d_input,
		T* d_k_cache,
		T* d_v_cache,
		int32_t* d_block_table,
		const int64_t block_num_per_layer,
		const int32_t* d_input_len,

		const int32_t* h_input_len,
		const bool* h_is_context_stage,
		const int64_t batch_size
	);

	void InitDecoder();

	void ForwardBlock(int64_t layer_id) override;

	void loadWeight(const std::string& model_path) override;
	void initDummyWeight() override;
	void initDummyWeightFSDP() override;
	std::vector<torch::Tensor>  GetFlattenWeight() override;
	void LazyInitWeight(bool enable_ipc_mem, bool is_alloc) override;

    std::vector<int64_t> forward(
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,
		void* d_k_cache,
		void* d_v_cache,
		int32_t* d_block_table,
		const int64_t block_num_per_layer
    ) override;

	void prologue(
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,
		torch::Tensor &k_cache,
		torch::Tensor &v_cache,
		int64_t** d_block_table
	) override;

	std::vector<int64_t> epilogue() override;

	std::vector<int64_t> pipelinedforward(
		const std::vector<std::vector<int64_t>> &input_tokens_batched,
		const std::vector<int64_t> &first_token_indexes,
		torch::Tensor &k_cache,
		torch::Tensor &v_cache,
		int64_t** d_block_table
	) override;
};

} // namespace st::model