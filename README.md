

# Semi-PD

A prefill & decode disaggregated LLM serving framework with shared GPU memory and fine-grained compute isolation.

## Acknowledgment
This repository originally started as a fork of the SGLang project. Semi-PD is a research prototype and does not have complete feature parity with open-source SGLang. We have only retained the most critical features and adopted the codebase for faster research iterations.

## Build && Install
```shell
# setup the semi-pd conda environment
conda env create -f semi_pd -y python=3.11
conda activate semi_pd

# Use the last release branch
git clone git@github.com:infinigence/Semi-PD.git
cd Semi-PD
pip install --upgrade pip

# build IPC dependency
cd ./semi-pd-ipc/
pip install -e .
```
### For NVIDIA GPUs
```shell
# build Semi-PD
cd ..
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
```
### For AMD GPUs
```shell
cd ../sgl-kernel
python setup_rocm.py install
cd ..
pip install -e "python[all_hip]"
```

## Using docker to build base environment

### Pull the NVIDIA image
```shell
docker pull lmsysorg/sglang:v0.4.4.post1-cu124
```

### Pull the AMD image
```shell
docker pull lmsysorg/sglang:v0.4.4.post1-rocm630
```

Then you can create a container from the image and follow the Build && Install section to build Semi-PD, or you can build from [Dockerfile](https://github.com/infinigence/Semi-PD/tree/update_readme/docker).




## Launching

### Introduce
The implementation of compute isolation is based on Multi-Process Service (MPS). For NVIDIA GPUs, the MPS service must be manually enabled, whereas on AMD GPUs, it is enabled by default.

### Enable MPS (NVIDIA)
```shell
export CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING=1
nvidia-cuda-mps-control -d
```

You can disable MPS service by using this cmd:
```shell
echo quit | sudo nvidia-cuda-mps-control
```

### Run online serving
Semi-PD can be enabled using the `--enable-semi-pd` flag. Additionally, our implementation does not share activations between the prefill and decode phases, which may result in slightly higher memory usage compared to the original SGLang. If an out-of-memory issue occurs, consider reducing the value of `--mem-fraction-static` to mitigate memory pressure.

```shell

python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --served-model-name $MODEL_NAME \
  --host 0.0.0.0 --port $SERVE_PORT --trust-remote-code  --disable-radix-cache \
  --enable-semi-pd  --mem-fraction-static 0.85
```

## Evaluation

![Semi-PD](./docs/_static/image/evaluation_semi_pd.png)
