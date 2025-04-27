import os
from dataclasses import dataclass
from enum import Enum
from typing import List

import semi_pd_ipc
import torch
import zmq

PREFILL_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_PREFILL_SM_PERCENTILE", 80))
DECODE_ENGINE_SM_PERCENTILE = int(os.getenv("SEMI_PD_DECODE_SM_PERCENTILE", 100))


@dataclass
class IPCInfo:
    params_info: dict
    weight_handles: dict
    register_buffer_handles: dict
    kv_cache_handles: list[list]
    kvcache_info: dict
    req_to_token_handle: list
    req_to_token_info: dict


class InstanceRole(Enum):
    PREFILL = 0
    DECODE = 1
    OTHER = 2


class AggregatedSocket:
    def __init__(self, sockets: List[zmq.Socket]):
        self.sockets = sockets

    def send_pyobj(self, obj):
        for socket in self.sockets:
            socket.send_pyobj(obj)


DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint64: "at::kUInt64",
    torch.uint32: "at::kUInt32",
    torch.uint16: "at::kUInt16",
    torch.uint8: "at::kByte",
    torch.uint32: "at::kUInt32",
    torch.uint64: "at::kUInt64",
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
    torch.complex32: "at::kComplexHalf",
    torch.complex64: "at::kComplexFloat",
    torch.complex128: "at::kComplexDouble",
    torch.float8_e4m3fn: "at::kFloat8_e4m3fn",
    torch.float8_e5m2: "at::kFloat8_e5m2",
    torch.float8_e4m3fnuz: "at::kFloat8_e4m3fnuz",
    torch.float8_e5m2fnuz: "at::kFloat8_e5m2fnuz",
}


def get_ipc_handle(tensor: torch.Tensor):
    # https://github.com/pytorch/pytorch/blob/cbcc03c2ad11fbf1080f6a1025cc3f7aee0c858d/torch/multiprocessing/reductions.py#L371
    (device, handle, storage_size_bytes# size(in bytes) of the storage
     , storage_offset_bytes# offset(in bytes) of the storage in the CUDA allocation
     ) = (
        tensor.storage()._share_cuda_()[:4]
    )
    assert storage_size_bytes == tensor.numel() * tensor.element_size()

    return semi_pd_ipc.get_ipc_handle(tensor), storage_offset_bytes


def convert_ipc_handle_to_tensor(ipc_handle, size, dtype, device):
    dtype_str = DTYPE_TO_ATEN[dtype]
    return semi_pd_ipc.convert_ipc_handle_to_tensor(ipc_handle, size, dtype_str, device)


def get_device_sm_count(rank: int = 0):
    return semi_pd_ipc.get_device_sm_count(rank)
