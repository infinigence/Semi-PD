#include <torch/script.h>

#include "util/py_nccl.h"
#include "util/py_swapping.h"
#include "util/py_block_migration.h"
#include "util/torch_utils.h"
#include "util/py_kernel.h"

#include "model/gpt/opt/optop.h"
#include "model/gpt/llama2/llama2op.h"
#include "model/gpt/gpt2/gpt2op.h"

/*
The two function wrappers below are needed to avoid the following error:

RuntimeError: Tried to convert an IValue of type __torch__.torch.classes.gpt_ops.OptOp (of Python compilation unit at: 0) to custom class type __torch__.torch.classes.gpt_ops.GptOpBase (of Python compilation unit at: 0)

We encounter the error above because of that, when we create a OptOp class in python and
call its load_weight() method, we are actually calling the load_weight() method of the
GptOpBase class, which is the base class of OptOp. However, PyTorch thinks it needs to
convert the OptOp object to a GptOpBase object (since the first argument of loadWeight
is GptOpBase*), which is not possible because we didn't defined that.

The solution is to define a wrapper function that takes a OptOp object as the first
argument and calls the loadWeight() method of the OptOp object, which avoid type
conversion.
*/
template<typename T>
void loadWeightWrapper(const c10::intrusive_ptr<T>& self, const std::string& path) {
  self->loadWeight(path);
}

template<typename T>
void initDummyWeightWrapper(const c10::intrusive_ptr<T>& self) {
  self->initDummyWeight();
}

template<typename T>
void initDummyWeightFSDPWrapper(const c10::intrusive_ptr<T>& self) {
  self->initDummyWeightFSDP();
}

template<typename T>
torch::Tensor GetCosSinCacheWrapper(const c10::intrusive_ptr<T>& self){
  return self->GetCosSinCache();
}

template<typename T>
std::vector<int64_t> forwardWrapper(const c10::intrusive_ptr<T>& self,
                                    const std::vector<std::vector<int64_t>> &input_tokens_batched,
                                    const std::vector<int64_t> &first_token_indexes,
                                    torch::Tensor &k_cache,
                                    torch::Tensor &v_cache,
                                    const std::vector<std::vector<int64_t>> &block_table) {
  return self->forward(input_tokens_batched, first_token_indexes, k_cache, v_cache, block_table);
}

template<typename T>
std::vector<int64_t> pipelinedforwardWrapper(const c10::intrusive_ptr<T>& self,
                                    const std::vector<std::vector<int64_t>> &input_tokens_batched,
                                    const std::vector<int64_t> &first_token_indexes,
                                    torch::Tensor &k_cache,
                                    torch::Tensor &v_cache,
                                    const std::vector<std::vector<int64_t>> &block_table) {
  return self->pipelinedforward(input_tokens_batched, first_token_indexes, k_cache, v_cache, block_table);
}

template<typename T>
void prologueWrapper(const c10::intrusive_ptr<T>& self,
                                    const std::vector<std::vector<int64_t>> &input_tokens_batched,
                                    const std::vector<int64_t> &first_token_indexes,
                                    torch::Tensor &k_cache,
                                    torch::Tensor &v_cache,
                                    const std::vector<std::vector<int64_t>> &block_table) {
  return self->prologue(input_tokens_batched, first_token_indexes, k_cache, v_cache, block_table);
}

template<typename T>
void ExecuteDecoderLayerWrapper(const c10::intrusive_ptr<T>& self,
                                    int64_t layer_id) {
  return self->ExecuteDecoderLayer(layer_id);
}

template<typename T>
std::vector<int64_t> epilogueWrapper(const c10::intrusive_ptr<T>& self) {
  return self->epilogue();
}

template<typename T>
void initCommunicatorWrapper(const c10::intrusive_ptr<T>& self,
                             const std::vector<int64_t> tp_id,
                             const std::vector<int64_t> pp_id) {
  self->init_communicator(tp_id, pp_id);
}

template<typename T>
void initP2PCommunicatorWrapper(const c10::intrusive_ptr<T>& self,
                             const std::vector<int64_t> peer_ids) {
  self->init_p2p_communicator(peer_ids);
}

template<typename T>
void SendWeightWrapper(const c10::intrusive_ptr<T>& self,
                             int64_t size) {
  self->send_weight(size);
}

template<typename T>
void RecvWeightWrapper(const c10::intrusive_ptr<T>& self,
                             int64_t size) {
  self->recv_weight(size);
}

template<typename T>
void WaitStreamWrapper(const c10::intrusive_ptr<T>& self) {
  self->wait_stream();
}

template<typename T>
void NcclGroupStartWrapper(const c10::intrusive_ptr<T>& self) {
  self->nccl_group_start();
}

template<typename T>
void NcclGroupEndWrapper(const c10::intrusive_ptr<T>& self) {
  self->nccl_group_end();
}

template<typename T>
std::vector<torch::Tensor> GetFlattenWeightWrapper(const c10::intrusive_ptr<T>& self) {
  return self->GetFlattenWeight();
}


template<typename T>
void LazyInitWeightWrapper(const c10::intrusive_ptr<T>& self, bool enable_ipc_mem, bool is_alloc) {
  self->LazyInitWeight(enable_ipc_mem, is_alloc);
}


TORCH_LIBRARY(gpt_ops, m) {
  m.class_<st::model::GptOpBase>("GptOpBase");  // Must add this class or will get error: "c10::intrusive_ptr<...> could not be converted to any of the known types."
  m.class_<st::model::OptOp>("OptOp")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::OptOp>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::OptOp>)
    .def("forward", &forwardWrapper<st::model::OptOp>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::OptOp>)
    .def("pipelined_forward", &pipelinedforwardWrapper<st::model::OptOp>)
    .def("prologue", &prologueWrapper<st::model::OptOp>)
    .def("execute_decoder_layer", &ExecuteDecoderLayerWrapper<st::model::OptOp>)
    .def("epilogue", &epilogueWrapper<st::model::OptOp>)
    .def("init_p2p_communicator", &initP2PCommunicatorWrapper<st::model::OptOp>)
    .def("send_weight", &SendWeightWrapper<st::model::OptOp>)
    .def("recv_weight", &RecvWeightWrapper<st::model::OptOp>)
    .def("wait_stream", &WaitStreamWrapper<st::model::OptOp>)
    .def("nccl_group_end", &NcclGroupEndWrapper<st::model::OptOp>)
    .def("nccl_group_start", &NcclGroupStartWrapper<st::model::OptOp>)
    .def("init_dummy_weights_fsdp", &initDummyWeightFSDPWrapper<st::model::OptOp>)
    .def("get_flatten_weight", &GetFlattenWeightWrapper<st::model::OptOp>)
    .def("lazy_init_weight", &LazyInitWeightWrapper<st::model::OptOp>)
    .def("get_cos_sin_cache", &GetCosSinCacheWrapper<st::model::OptOp>)

  ;
  m.class_<st::model::Llama2Op>("Llama2Op")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, bool, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::Llama2Op>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::Llama2Op>)
    .def("forward", &forwardWrapper<st::model::Llama2Op>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::Llama2Op>)
    .def("pipelined_forward", &pipelinedforwardWrapper<st::model::Llama2Op>)
    .def("prologue", &prologueWrapper<st::model::Llama2Op>)
    .def("execute_decoder_layer", &ExecuteDecoderLayerWrapper<st::model::Llama2Op>)
    .def("epilogue", &epilogueWrapper<st::model::Llama2Op>)
    .def("init_p2p_communicator", &initP2PCommunicatorWrapper<st::model::Llama2Op>)
    .def("send_weight", &SendWeightWrapper<st::model::Llama2Op>)
    .def("recv_weight", &RecvWeightWrapper<st::model::Llama2Op>)
    .def("wait_stream", &WaitStreamWrapper<st::model::Llama2Op>)
    .def("nccl_group_end", &NcclGroupEndWrapper<st::model::Llama2Op>)
    .def("nccl_group_start", &NcclGroupStartWrapper<st::model::Llama2Op>)
    .def("init_dummy_weights_fsdp", &initDummyWeightFSDPWrapper<st::model::Llama2Op>)
    .def("get_flatten_weight", &GetFlattenWeightWrapper<st::model::Llama2Op>)
    .def("lazy_init_weight", &LazyInitWeightWrapper<st::model::Llama2Op>)
    .def("get_cos_sin_cache", &GetCosSinCacheWrapper<st::model::Llama2Op>)


  ;
  m.class_<st::model::Gpt2Op>("Gpt2Op")
    .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t,
                     int64_t, std::string, int64_t, int64_t, std::vector<int64_t> >())
    .def("load_weight", &loadWeightWrapper<st::model::Gpt2Op>)
    .def("init_dummy_weights", &initDummyWeightWrapper<st::model::Gpt2Op>)
    .def("forward", &forwardWrapper<st::model::Gpt2Op>)
    .def("init_communicator", &initCommunicatorWrapper<st::model::Gpt2Op>)
    .def("pipelined_forward", &pipelinedforwardWrapper<st::model::Gpt2Op>)
    .def("prologue", &prologueWrapper<st::model::Gpt2Op>)
    .def("execute_decoder_layer", &ExecuteDecoderLayerWrapper<st::model::Gpt2Op>)
    .def("epilogue", &epilogueWrapper<st::model::Gpt2Op>)
    .def("init_p2p_communicator", &initP2PCommunicatorWrapper<st::model::Gpt2Op>)
    .def("send_weight", &SendWeightWrapper<st::model::Gpt2Op>)
    .def("recv_weight", &RecvWeightWrapper<st::model::Gpt2Op>)
    .def("wait_stream", &WaitStreamWrapper<st::model::Gpt2Op>)
    .def("nccl_group_end", &NcclGroupEndWrapper<st::model::Gpt2Op>)
    .def("nccl_group_start", &NcclGroupStartWrapper<st::model::Gpt2Op>)
    .def("init_dummy_weights_fsdp", &initDummyWeightFSDPWrapper<st::model::Gpt2Op>)
    .def("get_flatten_weight", &GetFlattenWeightWrapper<st::model::Gpt2Op>)
    .def("lazy_init_weight", &LazyInitWeightWrapper<st::model::Gpt2Op>)
    .def("get_cos_sin_cache", &GetCosSinCacheWrapper<st::model::Gpt2Op>)


  ;
}
TORCH_LIBRARY(nccl_ops, m)
{
    m.def("generate_nccl_id", &st::util::generate_nccl_id);
}

TORCH_LIBRARY(swapping_ops, m) {
  m.def("swap", &st::util::swap);
}

TORCH_LIBRARY(block_migration_ops, m) {
  m.def("get_ipc_mem_handle", &st::util::get_ipc_mem_handle);
  m.def("register_ipc_mem_handle", &st::util::register_ipc_mem_handle);
  m.def("migrate_blocks", &st::util::migrate_blocks);
  m.def("share_kv_cache_memory", &st::util::share_kv_cache_memory);
  m.def("register_weight_ipc_mem_handle", &st::util::register_weight_ipc_mem_handle);
  m.def("share_weight_memory", &st::util::share_weight_memory); 
  m.def("share_test_ipc_tensor", &st::util::share_test_ipc_tensor); 
  m.def("register_ipc_test_addr", &st::util::register_ipc_test_addr);
  m.def("get_device_available_SMs", &st::util::get_device_available_SMs);

}

TORCH_LIBRARY(custom_st_ops, m)
{
    m.def("forward_rope", &st::util::forwardRope);
    m.def("forward_rope_with_sin_cos_cache", &st::util::forwardRopeWithCosSinCache);
}