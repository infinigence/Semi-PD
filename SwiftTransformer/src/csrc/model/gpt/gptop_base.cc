#include "gptop_base.h"

#include "util/torch_utils.h"
#include "util/cuda_utils.h"
#include "util/nccl_utils.h"

namespace st::model {

GptOpBase::GptOpBase(
	std::string inference_dtype,
	GptHyperParam hyper_param,
	GptPagedAttnParam pagedattn_param,
	GptParallelismParam parallelism_param
) {
	if (inference_dtype == "fp32") {
		gpt = new Gpt<float>(hyper_param, pagedattn_param, parallelism_param);
	} else if (inference_dtype == "fp16") {
		gpt = new Gpt<__half>(hyper_param, pagedattn_param, parallelism_param);
	} else {
		throw std::runtime_error("Unsupported inference_dtype: " + inference_dtype);
	}

	weight_loaded = false;
}

GptOpBase::~GptOpBase() {
	delete gpt;
}

void GptOpBase::loadWeight(const std::string& weight_path) {
    this->gpt->loadWeight(weight_path);
    this->weight_loaded = true;
};

void GptOpBase::initDummyWeight() {
    this->gpt->initDummyWeight();
    this->weight_loaded = true;
};

void GptOpBase::initDummyWeightFSDP() {
    this->gpt->initDummyWeightFSDP();
    this->weight_loaded = true;
};

std::vector<torch::Tensor> GptOpBase::GetFlattenWeight(){
    return this->gpt->GetFlattenWeight();
}

void GptOpBase::LazyInitWeight(bool enable_ipc_mem, bool is_alloc){
    this->gpt->LazyInitWeight(enable_ipc_mem, is_alloc);
    this->weight_loaded = true;
}

torch::Tensor GptOpBase::GetCosSinCache(){
    return this->gpt->GetCosSinCache();
};

std::vector<int64_t> GptOpBase::forward(
    const std::vector<std::vector<int64_t>> &input_tokens_batched,
	const std::vector<int64_t> &first_token_indexes, // [batchsize]
    torch::Tensor &k_cache, // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor &v_cache, // [num_blocks, num_heads, block_size, head_dim]
    const std::vector<std::vector<int64_t>> &block_table)
{
    if (!this->weight_loaded) {
        throw std::runtime_error("Please load the weight before inference.");
    }

    int64_t batch_size = input_tokens_batched.size();
    if (batch_size == 0) {
		return std::vector<int64_t>();
    }

    // Prepare block_table
    // [hongke@1014] change it to INT32
    // int64_t* h_block_table = new int64_t[batch_size * this->gpt->pagedattn_param.max_num_block_per_req];
    // for (int64_t i = 0; i < batch_size; i++) {
    //     for (int64_t j = 0; j < (int64_t)block_table[i].size(); j++) {
    //         h_block_table[i * this->gpt->pagedattn_param.max_num_block_per_req + j] = block_table[i][j];
    //     }
    // }
    int32_t* h_block_table = new int32_t[batch_size * this->gpt->pagedattn_param.max_num_block_per_req];
    for (int32_t i = 0; i < batch_size; i++) {
        for (int32_t j = 0; j < (int32_t)block_table[i].size(); j++) {
            h_block_table[i * this->gpt->pagedattn_param.max_num_block_per_req + j] = (int32_t)block_table[i][j];
        }
    }

    // int64_t *d_block_table;
    // CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req));
    // CUDA_FREE_AT_RETURN(d_block_table);
    // cudaMemcpy(d_block_table, h_block_table, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);
    // [hongke@1014] change it to INT32
    int32_t *d_block_table;
    CUDA_CHECK(cudaMalloc(&d_block_table, sizeof(int32_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req));
    CUDA_FREE_AT_RETURN(d_block_table);
    cudaMemcpy(d_block_table, h_block_table, sizeof(int32_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);

    delete[] h_block_table;
    sync_check_cuda_error();

    const int64_t block_num_per_layer = k_cache.size(0);
    // fprintf(stderr, "block_num_per_layer \n %ld ", block_num_per_layer);

    return this->gpt->forward(input_tokens_batched,
                              first_token_indexes,
                              st::util::convertTensorToRawPtr(k_cache),
                              st::util::convertTensorToRawPtr(v_cache),
                              d_block_table,
                              // [hongke@1014] add block num per layer
                              block_num_per_layer
                              );
}

std::vector<int64_t> GptOpBase::pipelinedforward(
    const std::vector<std::vector<int64_t>> &input_tokens_batched,
	const std::vector<int64_t> &first_token_indexes, // [batchsize]
    torch::Tensor &k_cache, // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor &v_cache, // [num_blocks, num_heads, block_size, head_dim]
    const std::vector<std::vector<int64_t>> &block_table)
{
    if (!this->weight_loaded) {
        throw std::runtime_error("Please load the weight before inference.");
    }

    int64_t batch_size = input_tokens_batched.size();
    if (batch_size == 0) {
		return std::vector<int64_t>();
    }

    // Prepare block_table
    int64_t* h_block_table = new int64_t[batch_size * this->gpt->pagedattn_param.max_num_block_per_req];
    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t j = 0; j < (int64_t)block_table[i].size(); j++) {
            h_block_table[i * this->gpt->pagedattn_param.max_num_block_per_req + j] = block_table[i][j];
        }
    }
    int64_t *d_block_table_;
    CUDA_CHECK(cudaMalloc(&d_block_table_, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req));
    CUDA_FREE_AT_RETURN(d_block_table_);
    cudaMemcpy(d_block_table_, h_block_table, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);
    delete[] h_block_table;
    sync_check_cuda_error();

    return this->gpt->pipelinedforward(input_tokens_batched,
                              first_token_indexes,
                              k_cache,
                              v_cache,
                              &d_block_table_);
}

void GptOpBase::prologue(
    const std::vector<std::vector<int64_t>> &input_tokens_batched,
	const std::vector<int64_t> &first_token_indexes, // [batchsize]
    torch::Tensor &k_cache, // [num_blocks, num_heads, block_size, head_dim]
    torch::Tensor &v_cache, // [num_blocks, num_heads, block_size, head_dim]
    const std::vector<std::vector<int64_t>> &block_table)
{
    if (!this->weight_loaded) {
        throw std::runtime_error("Please load the weight before inference.");
    }

    int64_t batch_size = input_tokens_batched.size();
    if (batch_size == 0) {
		// return std::vector<int64_t>();
        early_return_ = true;
        return;
    }

    // Prepare block_table
    int64_t* h_block_table = new int64_t[batch_size * this->gpt->pagedattn_param.max_num_block_per_req];
    for (int64_t i = 0; i < batch_size; i++) {
        for (int64_t j = 0; j < (int64_t)block_table[i].size(); j++) {
            h_block_table[i * this->gpt->pagedattn_param.max_num_block_per_req + j] = block_table[i][j];
        }
    }
    int64_t *d_block_table_;
    CUDA_CHECK(cudaMalloc(&d_block_table_, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req));
    // CUDA_FREE_AT_RETURN(d_block_table_);
    cudaMemcpy(d_block_table_, h_block_table, sizeof(int64_t) * batch_size * this->gpt->pagedattn_param.max_num_block_per_req, cudaMemcpyHostToDevice);
    delete[] h_block_table;
    sync_check_cuda_error();

    this->gpt->prologue(input_tokens_batched, first_token_indexes, k_cache, v_cache, &d_block_table_);
    has_prologue_run_ = true;
}

void GptOpBase::ExecuteDecoderLayer(int64_t layer_id){
    if (early_return_ == true) return;
    if (has_prologue_run_ == false){
        throw std::runtime_error("Please run prologue first.");
    }
    this->gpt->ForwardBlock(layer_id);
}

std::vector<int64_t> GptOpBase::epilogue()
{
    if (early_return_ == true){
        early_return_ = false;
        return {};
    }
    auto output = this->gpt->epilogue();
    has_prologue_run_ = false;
    cudaFree(d_block_table_);
    cudaDeviceSynchronize();
    return output;
}


void GptOpBase::init_communicator(const std::vector<int64_t> tp_id, const std::vector<int64_t> pp_id){
    ncclUniqueId tp_uid, pp_uid;
	memcpy(tp_uid.internal, &tp_id[0], NCCL_UNIQUE_ID_BYTES);
	memcpy(pp_uid.internal, &pp_id[0], NCCL_UNIQUE_ID_BYTES);
    this->gpt->init_communicator(tp_uid, pp_uid);
}

void GptOpBase::init_p2p_communicator(const std::vector<int64_t> peer_ids){
    ncclUniqueId peer_id;
	memcpy(peer_id.internal, &peer_ids[0], NCCL_UNIQUE_ID_BYTES);
    this->gpt->init_p2p_communicator(peer_id);
}

void GptOpBase::send_weight(int64_t layer_id){
    this->gpt->send_weight(layer_id);
}

void GptOpBase::recv_weight(int64_t layer_id){
    this->gpt->recv_weight(layer_id);
}

void GptOpBase::wait_stream(){
    this->gpt->wait_stream();
}

void GptOpBase::nccl_group_start(){
    this->gpt->nccl_group_start();
}
void GptOpBase::nccl_group_end(){
    this->gpt->nccl_group_end();
}

}
