# Semi-PD distserve base

Semi-PD build from a fork of distserve https://github.com/LLMServe/DistServe.git.

## Build && Install
```shell
# setup the distserve conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build alioth
cd SwiftTransformer
git submodule update --init --recursive
cmake -B build && cmake --build build -j$(nproc)
cd ..

# install distserve
pip install -e .
```

## Launching

### Introduce

Currently we use ENABLE_MPS env variable to controll the switch between Semi-PD and DistServe.


### Enable MPS 
Only support on NV backend.

```shell
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
nvidia-cuda-mps-control -d
```

You can disable MPS service by using this cmd:
```shell
echo quit | sudo nvidia-cuda-mps-control
```


### Environment veriables
```shell
# Common
RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Semi-PD
ENABLE_MPS=1
CONTEXT_ENGINE_SM_PERCENTILE=100 # context instance sm usage percentile
DECODE_ENGINE_SM_PERCENTILE=100 # decode instance sm usage percentile

# DistServe
ENABLE_MPS=0

```

### Launch Ray Cluster

Semi-PD relies on [Ray](https://ray.io) to implement distributed workers. If you do not launch a Ray runtime in advance, it will automatically initiate a cluster consisting of all the gpus on the current node. You may need to start the Ray runtime manually in advance if you want to use multiple nodes for inference.

### Run offline example

Semi-PD requires at least 1 GPU to play with. We provide an offline inference example in `examples/offline.py`.

### Run online serving

```shell

ENABLE_MPS=1 CONTEXT_ENGINE_SM_PERCENTILE=100 DECODE_ENGINE_SM_PERCENTILE=100 ENABLE_MPS=1  python -m distserve.api_server.distserve_api_server \
    --host 0.0.0.0 \
    --model /path/to/model/ \
    --tokenizer /path/to/model/  \
    \
    --context-tensor-parallel-size 1 \
    --context-pipeline-parallel-size 1 \
    --decoding-tensor-parallel-size 1 \
    --decoding-pipeline-parallel-size 1 \
    \
    --block-size 16 \
    --max-num-blocks-per-req  384 \
    --gpu-memory-utilization 0.9 \
    --swap-space 16 \
    \
   --context-sched-policy fcfs \
    --context-max-batch-size 64 \
    --context-max-tokens-per-batch 4096 \
    \
    --decoding-sched-policy fcfs \
   --port 8000 \
    --decoding-max-batch-size 1024  --decoding-max-tokens-per-batch 200000 

```


