#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TORCH_EXTENSION_NAME semi_pd_ipc

// Serialize a CUDA IPC memory handle to a vector of bytes
static std::vector<int64_t> cudaIpcMemHandle2Bytes(const cudaIpcMemHandle_t &handle) {
	std::vector<int64_t> result;
	for (size_t i = 0; i < sizeof(handle); ++i) {
		result.push_back(((uint8_t*) &handle)[i]);
	}
	return result;
}

// Deserialize a vector of bytes to a CUDA IPC memory handle
static cudaIpcMemHandle_t bytes2CudaIpcMemHandle(const std::vector<int64_t> &bytes) {
	// assert_whenever(bytes.size() == sizeof(cudaIpcMemHandle_t));
	cudaIpcMemHandle_t result;
	for (size_t i = 0; i < sizeof(result); ++i) {
		((uint8_t*) &result)[i] = bytes[i];
	}
	return result;
}

at::ScalarType convertStringToDType(const std::string& aten_str) {
    static const std::unordered_map<std::string, at::ScalarType> ATEN_TO_DTYPE = {
        {"at::kFloat", at::kFloat},
        {"at::kDouble", at::kDouble},
        {"at::kHalf", at::kHalf},
        {"at::kLong", at::kLong},
        {"at::kInt", at::kInt},
        {"at::kShort", at::kShort},
        {"at::kChar", at::kChar},
        {"at::kByte", at::kByte},
        {"at::kUInt64", at::kUInt64},
        {"at::kUInt32", at::kUInt32},
        {"at::kUInt16", at::kUInt16},
        {"at::kBool", at::kBool},
        {"at::kBFloat16", at::kBFloat16},
        {"at::kComplexHalf", at::kComplexHalf},
        {"at::kComplexFloat", at::kComplexFloat},
        {"at::kComplexDouble", at::kComplexDouble},
        {"at::kFloat8_e4m3fn", at::kFloat8_e4m3fn},
        {"at::kFloat8_e5m2", at::kFloat8_e5m2},
        {"at::kFloat8_e4m3fnuz", at::kFloat8_e4m3fnuz},
        {"at::kFloat8_e5m2fnuz", at::kFloat8_e5m2fnuz},
    };

    auto it = ATEN_TO_DTYPE.find(aten_str);
    if (it != ATEN_TO_DTYPE.end()) {
        return it->second;
    }
    throw std::invalid_argument("Unsupported at::ScalarType string: " + aten_str);
}

// Convert a CUDA tensor to a CUDA IPC memory handle
std::vector<int64_t> GetIPCMemHandle(torch::Tensor tensor) {
	cudaIpcMemHandle_t handle;
	cudaIpcGetMemHandle(&handle, tensor.data_ptr());
	return cudaIpcMemHandle2Bytes(handle);
}

// Convert a CUDA IPC memory handle to a CUDA tensor
torch::Tensor ConvertIPCMemHandleToTensor(std::tuple<std::vector<int64_t>, uint64_t> handle_vec_offset, int64_t tensor_size, std::string dtype_str, torch::Device device) {
    // Convert the handles to cudaIpcMemHandle_t
    auto const& [handle_vec, offsets] = handle_vec_offset;

	const cudaIpcMemHandle_t handle = bytes2CudaIpcMemHandle(handle_vec);
    // Open the memory handle
	void* ipc_addr;
	cudaError_t err = cudaIpcOpenMemHandle(&ipc_addr, handle, cudaIpcMemLazyEnablePeerAccess);
	if (err == cudaErrorPeerAccessUnsupported) {
		printf("Error: Peer-to-peer access is unsupported on this platform.\n");
		printf("In the current version of distserve, it is necessary to use a platform that supports GPU P2P access.\n");
		printf("Exiting...");
		exit(1);
	}
    uint8_t* real_ipc_addr = static_cast<uint8_t*>(ipc_addr) + offsets;
	torch::Tensor ipc_tensor = torch::from_blob(real_ipc_addr, tensor_size,
                          	torch::TensorOptions().dtype(convertStringToDType(dtype_str)).device(device));
    return ipc_tensor;
}

int64_t GetDeviceSMCount(int rank){
	cudaSetDevice(rank);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, rank);
	return deviceProp.multiProcessorCount;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_ipc_handle", &GetIPCMemHandle, "Convert a CUDA tensor to a CUDA IPC memory handle");
    m.def("convert_ipc_handle_to_tensor", &ConvertIPCMemHandleToTensor, "Convert a CUDA IPC memory handle to a CUDA tensor");
    m.def("get_device_sm_count", &GetDeviceSMCount, "Get the number of available SMs on a given device");
}
