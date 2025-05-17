#pragma once

#include <iostream>
#include <vector>

#include "model/gpt/gpt_hyper_param.h"

namespace st::model {

struct GptParallelismParam {
    // Hyper parameters related to parallelism
    int64_t tensor_para_size = 1;
    int64_t tensor_para_rank = 0;

    int64_t pipeline_para_size = 1;
    int64_t pipeline_para_rank = 0;

    int64_t is_context = 0;

    bool hyper_inited = false;

    // The following two parameters are used for pipeline parallelism
    // The layer range of the current pipeline stage is [layer_begin, layer_end)
    int64_t layer_begin = 0, layer_end = 0, local_layer_num = 0;

    GptParallelismParam(int64_t tensor_para_size = 1, int64_t tensor_para_rank = 0, int64_t pipeline_para_size = 1, int64_t pipeline_para_rank = 0, int64_t is_context = 0)
        : tensor_para_size(tensor_para_size)
        , tensor_para_rank(tensor_para_rank)
        , pipeline_para_size(pipeline_para_size)
        , pipeline_para_rank(pipeline_para_rank)
        , is_context(is_context)
    {
    }

    GptParallelismParam(const std::vector<int64_t> parallel_config)
        : GptParallelismParam(parallel_config[0], parallel_config[1], parallel_config[2], parallel_config[3], parallel_config[4])
    {
    }

    void init_by_hyper_param(const GptHyperParam& hyper_param)
    {      
        if (hyper_inited) {
            return;
        }
        hyper_inited = true;
        if (hyper_param.num_layers % pipeline_para_size == 0)
        {
            local_layer_num = hyper_param.num_layers / pipeline_para_size;
            layer_begin = pipeline_para_rank * local_layer_num;
            layer_end = layer_begin + local_layer_num;
        } else {
            int64_t base_layer_num = hyper_param.num_layers / pipeline_para_size;
            int64_t gap_layer_num = hyper_param.num_layers - pipeline_para_size * base_layer_num;
            if (pipeline_para_rank < gap_layer_num){
                local_layer_num = base_layer_num + 1;
                layer_begin = pipeline_para_rank * local_layer_num;
                layer_end = layer_begin + local_layer_num;
            } else {
                local_layer_num = base_layer_num;
                layer_begin = gap_layer_num * (base_layer_num + 1) + (pipeline_para_rank - gap_layer_num) * base_layer_num;
                layer_end = layer_begin + local_layer_num;
            }
            fprintf(stderr, "The number of layers cannot be divided by the pipeline parallelism size, cur pipeline stage %ld, layer nums %ld\n", pipeline_para_rank, local_layer_num);
        }
        
    }

    inline bool is_parallel() const
    {
        return tensor_para_size > 1 || pipeline_para_size > 1;
    }

    inline bool is_last_stage() const
    {
        return pipeline_para_rank == pipeline_para_size - 1;
    }

    inline bool is_first_stage() const
    {
        return pipeline_para_rank == 0;
    }

    inline bool is_stage_leader() const
    {
        return tensor_para_rank == 0;
    }

    void set_parallelism(int64_t tensor_para_size, int64_t tensor_para_rank, int64_t pipeline_para_size, int64_t pipeline_para_rank)
    {
        this->tensor_para_size = tensor_para_size;
        this->tensor_para_rank = tensor_para_rank;
        this->pipeline_para_size = pipeline_para_size;
        this->pipeline_para_rank = pipeline_para_rank;
    }

    friend std::ostream& operator<<(std::ostream& os, const GptParallelismParam& param)
    {
        os << "tensor_para_size: " << param.tensor_para_size << std::endl;
        os << "tensor_para_rank: " << param.tensor_para_rank << std::endl;
        os << "pipeline_para_size: " << param.pipeline_para_size << std::endl;
        os << "pipeline_para_rank: " << param.pipeline_para_rank << std::endl;
        os << "is_context: " << param.is_context << std::endl;
        return os;
    }
};

} // namespace st::model
