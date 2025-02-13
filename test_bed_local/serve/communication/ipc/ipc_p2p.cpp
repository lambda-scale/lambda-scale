#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <iostream>
#include <chrono>
#include <c10/core/StorageImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <unordered_map>
#include <string>
#include <fcntl.h>  // For open
#include <unistd.h> // For pwrite
#include <stdexcept>
#include <cstring>

namespace py = pybind11;

void checkCudaError(cudaError_t result, const char *func) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " at " << func << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val)

torch::ScalarType get_dtype_from_element_size(int element_size) {
    if (element_size == 1) {
        return torch::kInt8;
    } else if (element_size == 2) {
        return torch::kFloat16;
    } else if (element_size == 4) {
        return torch::kFloat32;
    } else {
        throw std::invalid_argument("Unsupported element size. Only 1, 2, and 4 are supported.");
    }
}

// py::tuple create_tensor_with_meta_data(const std::vector<int64_t>& shape, int element_size, int device_id) {
//     cudaSetDevice(device_id);

//     torch::ScalarType dtype = get_dtype_from_element_size(element_size);

//     int64_t numel = 1;
//     for (auto dim : shape) {
//         numel *= dim;
//     }
//     int64_t num_bytes = numel * element_size;

//     void* tensor_data;
//     CHECK_CUDA_ERROR(cudaMalloc(&tensor_data, num_bytes));

//     auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
//     torch::Tensor tensor = at::empty(shape, options);
//     tensor.storage().set_data_ptr(c10::DataPtr(reinterpret_cast<void*>(tensor_data), c10::Device(c10::DeviceType::CUDA, device_id)));

//     // Create IPC handle for the memory
//     cudaIpcMemHandle_t ipc_handle;
//     cudaIpcGetMemHandle(&ipc_handle, tensor_data);

//     // Convert IPC handle to bytes for transfer
//     py::bytes ipc_handle_bytes(reinterpret_cast<char*>(&ipc_handle), sizeof(cudaIpcMemHandle_t));

//     return py::make_tuple(tensor, tensor_data,ipc_handle_bytes,num_bytes);
// }


// void copy_to_existing_tensor(const std::string& handle, int64_t size, int64_t offset, int src_device_id, int dst_device_id, torch::Tensor& dst_tensor, int64_t stream_ptr) {
//   // 设置源 GPU 设备
//   cudaSetDevice(src_device_id);

//   // 处理存储句柄
//   cudaIpcMemHandle_t ipc_handle;
//   memcpy(&ipc_handle, handle.data(), sizeof(cudaIpcMemHandle_t));

//   void* dev_ptr;
//   CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(&dev_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));


//   CHECK_CUDA_ERROR(cudaSetDevice(dst_device_id));


//   int64_t num_bytes = size * dst_tensor.element_size();

// //   cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
//   cudaStream_t stream;
//   CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

// //   std::cout << "Source Device ID: " << src_device_id << std::endl;
// //     std::cout << "Destination Device ID: " << dst_device_id << std::endl;
// //     std::cout << "Device Pointer (original): " << reinterpret_cast<void*>(dev_ptr) << std::endl;
// //     std::cout << "Destination Tensor Pointer: " << dst_tensor.data_ptr() << std::endl;
// //     std::cout << "Offset: " << offset << std::endl;
// //     std::cout << "Number of Bytes: " << num_bytes << std::endl;
//   CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(dst_tensor.data_ptr(), dst_device_id,static_cast<float*>(dev_ptr) + offset,src_device_id, num_bytes,stream));
//   CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

//   cudaSetDevice(src_device_id);
//   cudaIpcCloseMemHandle(dev_ptr);


//   cudaSetDevice(dst_device_id);
// }

void gpu_unmalloc(uintptr_t ptr) {
    void* dev_ptr = reinterpret_cast<void*>(ptr);
    CHECK_CUDA_ERROR(cudaFree(dev_ptr));
}

void cpu_unmalloc(uintptr_t ptr) {
    CHECK_CUDA_ERROR(cudaFreeHost(reinterpret_cast<void*>(ptr)));
    // void* cpu_ptr = reinterpret_cast<void*>(ptr);
    // free(cpu_ptr);  
}

py::tuple gpu_allocate_memory_and_get_ipc_handle(int64_t num_bytes ,int device_id) {
    void* dev_ptr;
    cudaSetDevice(device_id);
    cudaError_t result = cudaMalloc(&dev_ptr, num_bytes);
    checkCudaError(result, "cudaMalloc");

    // 创建 IPC 句柄
    cudaIpcMemHandle_t ipc_handle;
    CHECK_CUDA_ERROR(cudaIpcGetMemHandle(&ipc_handle, dev_ptr));
    // 将句柄转换为字节数组传递给 Python
    py::bytes handle(reinterpret_cast<const char*>(&ipc_handle), sizeof(cudaIpcMemHandle_t));

    // 返回内存指针和 IPC 句柄作为 Python 元组
    return py::make_tuple(reinterpret_cast<uintptr_t>(dev_ptr), handle);
}

torch::Tensor gpu_create_tensor_from_ptr(uintptr_t ptr, std::vector<int64_t> shape, int element_size ,int device_id) {
    cudaSetDevice(device_id);
    torch::ScalarType dtype;
    if (element_size == 1) {
        dtype = torch::kInt8;   
    } else if (element_size == 2) {
        dtype = torch::kFloat16;  
    } else if (element_size == 4) {
        dtype = torch::kFloat32;  
    } else {
        throw std::invalid_argument("Unsupported element size. Only 1, 2, and 4 are supported.");
    }

    // auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    // torch::Tensor tensor = at::empty(shape, options);
    // tensor.storage().set_data_ptr(c10::DataPtr(reinterpret_cast<void*>(ptr), c10::Device(c10::DeviceType::CUDA, device_id)));

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    void* void_ptr = reinterpret_cast<void*>(ptr); 
    torch::Tensor tensor = torch::from_blob(void_ptr, shape, options);
    return tensor;
}

torch::Tensor gpu_create_1dtensor(uintptr_t ptr, size_t bytes, int device_id) {
    cudaSetDevice(device_id);

    auto dtype = torch::kUInt8;

    int64_t length = bytes;

    // auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    // torch::Tensor tensor = at::empty({length}, options);

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    void* void_ptr = reinterpret_cast<void*>(ptr); 
    torch::Tensor tensor = torch::from_blob(void_ptr, {length}, options);
    // tensor.storage().set_data_ptr(c10::DataPtr(reinterpret_cast<void*>(ptr), c10::Device(c10::DeviceType::CUDA, device_id)));

    return tensor;
}


uintptr_t cpu_allocate_memory(int64_t num_bytes) {
    void* cpu_ptr;
    CHECK_CUDA_ERROR(cudaHostAlloc(&cpu_ptr, num_bytes, cudaHostAllocDefault));
    // void* cpu_ptr = malloc(num_bytes);
    // if (!cpu_ptr) {
    //     throw std::runtime_error("Failed to allocate CPU memory");
    // }

    return reinterpret_cast<uintptr_t>(cpu_ptr);
}

torch::Tensor cpu_create_tensor_from_ptr(uintptr_t ptr, std::vector<int64_t> shape, int element_size) {
    torch::ScalarType dtype;
    if (element_size == 1) {
        dtype = torch::kInt8;   
    } else if (element_size == 2) {
        dtype = torch::kFloat16;  
    } else if (element_size == 4) {
        dtype = torch::kFloat32;
    } else {
        throw std::invalid_argument("Unsupported element size. Only 1, 2, and 4 are supported.");
    }

    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
    // torch::Tensor tensor = at::empty(shape, options);
    // tensor.storage().set_data_ptr(c10::DataPtr(reinterpret_cast<void*>(ptr), c10::Device(c10::DeviceType::CPU)));

    void* void_ptr = reinterpret_cast<void*>(ptr);
    torch::Tensor tensor = torch::from_blob(void_ptr, shape, options);
    
    return tensor;
}


uintptr_t open_mem_handle(py::bytes remote_handle_bytes,int device_id){
    std::string remote_handle_str = remote_handle_bytes;
    // 将字符串转换回 cudaIpcMemHandle_t
    cudaIpcMemHandle_t ipc_handle;
    memcpy(&ipc_handle, remote_handle_str.data(), sizeof(cudaIpcMemHandle_t));

    cudaSetDevice(device_id);

    void* ptr;
    CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));

    return reinterpret_cast<uintptr_t>(ptr);
}

void close_mem_handle(uintptr_t ptr,int device_id){
    cudaSetDevice(device_id);
    CHECK_CUDA_ERROR(cudaIpcCloseMemHandle(reinterpret_cast<void*>(ptr)));
}

void inner_node_transfer_data(uintptr_t local_ptr,int local_device_id,uintptr_t remote_ptr,int remote_device_id,int64_t num_bytes){
    cudaSetDevice(local_device_id);
    CHECK_CUDA_ERROR(cudaMemcpyPeer(reinterpret_cast<void*>(local_ptr),local_device_id,reinterpret_cast<void*>(remote_ptr),remote_device_id, static_cast<size_t>(num_bytes)));
}

void read_from_remote_handle(py::bytes remote_handle_bytes, int64_t num_bytes, int local_device_id, int remote_device_id, uintptr_t local_ptr) {

    auto start = std::chrono::high_resolution_clock::now();

    std::string remote_handle_str = remote_handle_bytes;

    // 将字符串转换回 cudaIpcMemHandle_t
    cudaIpcMemHandle_t ipc_handle;
    memcpy(&ipc_handle, remote_handle_str.data(), sizeof(cudaIpcMemHandle_t));
    cudaSetDevice(remote_device_id);

    void* remote_ptr;
    CHECK_CUDA_ERROR(cudaIpcOpenMemHandle(&remote_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess));

    auto stop = std::chrono::high_resolution_clock::now();
    // 计算持续时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();

    cudaSetDevice(local_device_id);

    CHECK_CUDA_ERROR(cudaMemcpyPeer(reinterpret_cast<void*>(local_ptr),local_device_id, remote_ptr,remote_device_id, static_cast<size_t>(num_bytes)));

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;


    start = std::chrono::high_resolution_clock::now();

    cudaIpcCloseMemHandle(remote_ptr);


    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;


    // CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    // CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    // float milliseconds = 0;
    // CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    // std::cout << "memcpy " << milliseconds << " ms" << std::endl;
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "all Time taken: " << duration.count() << " microseconds" << std::endl;
}

void copy_from_tensor_to_tensor(torch::Tensor& tensor1,torch::Tensor& tensor2) {
    int64_t bytes = tensor1.numel()*tensor1.element_size();

    // 获取 Tensor 的数据指针
    void* data_ptr1 = tensor1.data_ptr();
    void* data_ptr2 = tensor2.data_ptr();
    CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(data_ptr1), 
                                 reinterpret_cast<void*>(data_ptr2), 
                                 bytes, 
                                 cudaMemcpyDeviceToDevice));
    return;
}
void copy_from_tensor_to_tensor_async(torch::Tensor& tensor1, int dst_gpu, torch::Tensor& tensor2, int src_gpu ,int64_t stream_ptr) {
    int64_t bytes = tensor1.numel() * tensor1.element_size();

    // Get the data pointers for both tensors.
    void* data_ptr1 = tensor1.data_ptr();
    void* data_ptr2 = tensor2.data_ptr();
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    CHECK_CUDA_ERROR(cudaSetDevice(dst_gpu));

    CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(reinterpret_cast<void*>(data_ptr1), 
                                 dst_gpu,
                                 reinterpret_cast<void*>(data_ptr2), 
                                 src_gpu,
                                 bytes,
                                 stream));
}

void inner_node_transfer_data_async(uintptr_t local_ptr,int local_device_id,uintptr_t remote_ptr,int remote_device_id,int64_t num_bytes,int64_t stream_ptr){
    cudaSetDevice(local_device_id);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    CHECK_CUDA_ERROR(cudaMemcpyPeerAsync(reinterpret_cast<void*>(local_ptr),
    local_device_id,
    reinterpret_cast<void*>(remote_ptr),
    remote_device_id, 
    static_cast<size_t>(num_bytes),
    stream
    ));
}

void copy_tensor_from_memory_to_gpu(torch::Tensor& tensor,uintptr_t gpu_ptr,size_t bytes) {
    void* cpu_ptr = tensor.data_ptr();

    // assert !!!
    // size_t size = param.nbytes();

    CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(gpu_ptr), 
                                 reinterpret_cast<void*>(cpu_ptr), 
                                 bytes, 
                                 cudaMemcpyHostToDevice));
    return;
}

void copy_from_memory_to_gpu(uintptr_t gpu_ptr,uintptr_t cpu_ptr,size_t bytes){
    CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(gpu_ptr), 
                                 reinterpret_cast<void*>(cpu_ptr), 
                                 bytes, 
                                 cudaMemcpyHostToDevice));
    return;
}

void copy_from_memory_to_gpu_async(uintptr_t gpu_ptr,uintptr_t cpu_ptr,size_t bytes,int64_t stream_ptr){
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<void*>(gpu_ptr), 
                                 reinterpret_cast<void*>(cpu_ptr), 
                                 bytes, 
                                 cudaMemcpyHostToDevice,
                                 stream));
    return;
}


void copy_from_gpu_to_memory(uintptr_t cpu_ptr,uintptr_t gpu_ptr,size_t bytes){
    CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(cpu_ptr), 
                                 reinterpret_cast<void*>(gpu_ptr), 
                                 bytes, 
                                 cudaMemcpyDeviceToHost));
    return;
}



void copy_from_gpu_to_memory_async(uintptr_t cpu_ptr,uintptr_t gpu_ptr,size_t bytes,int64_t stream_ptr){
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(reinterpret_cast<void*>(cpu_ptr),
                                 reinterpret_cast<void*>(gpu_ptr), 
                                 bytes, 
                                 cudaMemcpyDeviceToHost,
                                 stream));
    return;
}

void copy_from_gpu_to_gpu(uintptr_t gpu_ptr1,uintptr_t gpu_ptr2,size_t bytes){
    CHECK_CUDA_ERROR(cudaMemcpy(reinterpret_cast<void*>(gpu_ptr1), 
                                 reinterpret_cast<void*>(gpu_ptr2), 
                                 bytes, 
                                 cudaMemcpyDeviceToDevice));
    return;
}

void read_from_ssd_to_cpu(const std::string& file_path, uintptr_t cpu_ptr, size_t bytes_to_read) {
    size_t chunk_size = 1024 * 1024 * 1024;
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    size_t bytes_read = 0;
    while (bytes_read < bytes_to_read) {
        size_t remaining_bytes = bytes_to_read - bytes_read;
        size_t current_chunk_size = (remaining_bytes < chunk_size) ? remaining_bytes : chunk_size;

        ssize_t result = pread(fd, reinterpret_cast<void*>(cpu_ptr + bytes_read), current_chunk_size, bytes_read);
        if (result == -1) {
            close(fd);  
            throw std::runtime_error("Failed to read from file: " + file_path);
        }

        bytes_read += result;

        if (result < current_chunk_size) {
            std::cerr << "Warning: Only read " << result << " bytes out of " << current_chunk_size << " bytes requested." << std::endl;
            break;
        }
    }

    close(fd);
}

void write_from_cpu_to_ssd(const std::string& file_path, uintptr_t cpu_ptr, size_t bytes_to_write) {
    // Open file with appropriate flags and permissions
    int fd = open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        throw std::runtime_error("Failed to open or create file: " + file_path);
    }

    // Maximum chunk size for each pwrite call (e.g., 1GB = 1073741824 bytes)
    const size_t CHUNK_SIZE = 1073741824;  // 1 GB
    size_t bytes_written = 0;

    while (bytes_written < bytes_to_write) {
        // Calculate size for the next chunk
        size_t chunk = std::min(CHUNK_SIZE, bytes_to_write - bytes_written);

        // Perform the write operation for the current chunk
        ssize_t result = pwrite(fd, reinterpret_cast<void*>(cpu_ptr + bytes_written), chunk, bytes_written);
        if (result == -1) {
            close(fd);
            throw std::runtime_error("Failed to write to file: " + file_path);
        }

        // Update the number of bytes written
        bytes_written += result;
    }

    // Close the file descriptor
    close(fd);
}

void warm_up(int gpu_num, size_t num_bytes) {
    if (gpu_num < 2) {
        std::cerr << "Need at least 2 GPUs for warm-up." << std::endl;
        return;
    }
    
    for (int i = 0; i < gpu_num-1; ++i) {
        for (int j = i+1; j< gpu_num ; ++j){
            int canAccessPeer = 0;
            CHECK_CUDA_ERROR(cudaDeviceCanAccessPeer(&canAccessPeer, i, j));
            if (canAccessPeer) {
                CHECK_CUDA_ERROR(cudaSetDevice(i));
                CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(j, 0));

                CHECK_CUDA_ERROR(cudaSetDevice(j));
                CHECK_CUDA_ERROR(cudaDeviceEnablePeerAccess(i, 0));
            } else {
                std::cerr << "P2P not supported between device " << i << " and device " << j<< "." << std::endl;
            }
        }
    }

    // 分配内存指针数组，用于存储在每个 GPU 上分配的内存
    std::vector<void*> d_ptrs(gpu_num);

    // 在每个 GPU 上分配内存
    for (int i = 0; i < gpu_num; ++i) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaMalloc(&d_ptrs[i], num_bytes));
    }

    // 执行预热：在 GPU 之间进行多次数据拷贝
    for (int i = 0; i < gpu_num - 1; ++i) {
        int src_gpu = i;
        int dst_gpu = i + 1;

        CHECK_CUDA_ERROR(cudaSetDevice(dst_gpu));
        CHECK_CUDA_ERROR(cudaMemcpyPeer(d_ptrs[dst_gpu], dst_gpu, d_ptrs[src_gpu], src_gpu, num_bytes));
    }

    // 清理资源
    for (int i = 0; i < gpu_num; ++i) {
        CHECK_CUDA_ERROR(cudaSetDevice(i));
        CHECK_CUDA_ERROR(cudaFree(d_ptrs[i]));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("copy_tensor_from_memory_to_gpu",&copy_tensor_from_memory_to_gpu,"copy_tensor_from_memory_to_gpu");
  m.def("copy_from_memory_to_gpu",&copy_from_memory_to_gpu,"copy_from_memory_to_gpu");
  m.def("copy_from_gpu_to_memory",&copy_from_gpu_to_memory,"copy_from_gpu_to_memory");
  m.def("copy_from_gpu_to_gpu",&copy_from_gpu_to_gpu,"copy_from_gpu_to_gpu");
  m.def("copy_from_tensor_to_tensor",&copy_from_tensor_to_tensor,"copy_from_tensor_to_tensor");
  m.def("copy_from_tensor_to_tensor_async",&copy_from_tensor_to_tensor_async,"copy_from_tensor_to_tensor_async");
  m.def("copy_from_gpu_to_memory_async",&copy_from_gpu_to_memory_async,"copy_from_gpu_to_memory_async");
  m.def("copy_from_memory_to_gpu_async",&copy_from_memory_to_gpu_async,"copy_from_memory_to_gpu_async");

  m.def("read_from_ssd_to_cpu",&read_from_ssd_to_cpu,"read_from_ssd_to_cpu");
  m.def("write_from_cpu_to_ssd",&write_from_cpu_to_ssd,"write_from_cpu_to_ssd");
  
  m.def("open_mem_handle",&open_mem_handle,"open_mem_handle");
//   m.def("create_and_copy_tensor",&create_and_copy_tensor,"create_and_copy_tensor");
//   m.def("create_tensor_with_meta_data",&create_tensor_with_meta_data,"create_tensor_with_meta_data");
//   m.def("copy_to_existing_tensor", &copy_to_existing_tensor, "get");

  m.def("gpu_unmalloc",&gpu_unmalloc,"gpu_unmalloc");
  m.def("cpu_unmalloc",&cpu_unmalloc,"cpu_unmalloc");
  m.def("gpu_allocate_memory_and_get_ipc_handle", &gpu_allocate_memory_and_get_ipc_handle,py::return_value_policy::move,"Allocate CUDA memory and return a pointer");
  m.def("gpu_create_tensor_from_ptr", &gpu_create_tensor_from_ptr, "Create a PyTorch tensor from an existing CUDA memory pointer");
  m.def("gpu_create_1dtensor", &gpu_create_1dtensor, "Create a PyTorch tensor from an existing CUDA memory pointer");
  m.def("cpu_allocate_memory", &cpu_allocate_memory, "Create a PyTorch tensor from an existing CUDA memory pointer");
  m.def("cpu_create_tensor_from_ptr", &cpu_create_tensor_from_ptr, "Create a PyTorch tensor from an existing CUDA memory pointer");
  
  m.def("close_mem_handle", &close_mem_handle, "close_mem_handle");
  m.def("inner_node_transfer_data",&inner_node_transfer_data,"inner_node_transfer_data");
  m.def("inner_node_transfer_data_async",&inner_node_transfer_data_async,"inner_node_transfer_data_async");
  m.def("read_from_remote_handle", &read_from_remote_handle, "Read data from a remote IPC handle into a local CUDA memory pointer");
  m.def("warm_up", &warm_up, "Warm up the GPUs by performing cudaMemcpy between them");
}