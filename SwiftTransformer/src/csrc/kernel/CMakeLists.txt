add_library(kernel STATIC
	addbias.cu
	count_nan.cu
	embedding.cu
	findmax.cu
	fused_activ_multiply.cu
	fused_addbias_activ.cu
	fused_context_stage_attention.cu
	fused_decoding_stage_attention.cu
	fused_decoding_stage_attention_mha.cu
	gather_last_tokens.cu
	kvcache_mgmt.cu
	layernorm.cu
	rmsnorm.cu
	rotary_posi_embedding.cu
	softmax.cu
	unfused_attention.cu
)
target_link_libraries(kernel PUBLIC util)
set_property(TARGET kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB xformers_autogen_impl_files ${CMAKE_CURRENT_SOURCE_DIR}/xformers/xformers/csrc/attention/cuda/fmha/autogen/impl/*.cu)
add_library(xformers_autogen_impl STATIC ${xformers_autogen_impl_files})
target_include_directories(xformers_autogen_impl PUBLIC xformers/third_party/cutlass/include)
set_property(TARGET xformers_autogen_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(xformers_kernel STATIC
	xformers_attention.cu
)
target_include_directories(xformers_kernel PUBLIC xformers/third_party/cutlass/include)
target_link_libraries(xformers_kernel PUBLIC util xformers_autogen_impl)
set_property(TARGET xformers_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

file(GLOB flash_autogen_impl_files ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash-attention/csrc/flash_attn/src/flash_fwd*.cu)
add_library(flash_autogen_impl STATIC ${flash_autogen_impl_files})
target_include_directories(flash_autogen_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash-attention/csrc/cutlass/include)
target_include_directories(flash_autogen_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash-attention/csrc/cutlass/tools/util/include)
target_include_directories(flash_autogen_impl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash-attention/csrc/flash_attn/src)
set_property(TARGET flash_autogen_impl PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(flash_kernel STATIC
 flash_attn/flash_attn.cu
 flash_attn/flash_api.cpp
)
target_include_directories(flash_kernel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/flash_attn/flash-attention/csrc/cutlass/include)
target_link_libraries(flash_kernel PUBLIC util flash_autogen_impl)
# target_link_libraries(flash_kernel PUBLIC /share/hongke/anaconda3/envs/distserve/lib/libpython3.10.so)
set_property(TARGET flash_kernel PROPERTY POSITION_INDEPENDENT_CODE ON)

