from test_bed_local.serve.model_info.model_info import ModelInfo, ModelStorageStructure
from test_bed_local.serve.model_info.model_loader import load_model_by_name
from test_bed_local.serve.server.model_storage.model_storage import save_model
import ipc_p2p

root_path = '/jiachaobo/test'
model_name = 'llama-2-13b'
model_info = ModelInfo(model_name,
                            root_path=root_path)
gpu_id = 0

model = load_model_by_name(model_name,gpu_id,root_path)
model_storage_structure : ModelStorageStructure =  ModelStorageStructure(model_name=model_name,
                                                                                    root_path=root_path)

gpu_ptrs = []
for block_id in range(model_info.get_block_num()):
    gpu_ptr,handle = ipc_p2p.gpu_allocate_memory_and_get_ipc_handle(model_storage_structure.block_storage_bytes_list[block_id], gpu_id)
    gpu_ptrs.append(gpu_ptr)

model_storage_structure.model_redirect_same_process(
    gpu_ptrs = gpu_ptrs,
    device_id = gpu_id,
    model = model,
    is_init = True
)

cpu_ptrs = []
for block_id in range(model_info.get_block_num()):
    cpu_ptrs.append(ipc_p2p.cpu_allocate_memory(model_storage_structure.block_storage_bytes_list[block_id]))
    ipc_p2p.copy_from_gpu_to_memory(cpu_ptrs[block_id],gpu_ptrs[block_id],model_storage_structure.block_storage_bytes_list[block_id])
    file_path = f'{model_info.root_path}/gpu-fast-scaling/test_bed_local/serve/server/model_storage/{model_info.model_name}/{block_id}.pth'
    print(model_storage_structure.block_storage_bytes_list[block_id])
    ipc_p2p.write_from_cpu_to_ssd(file_path, cpu_ptrs[block_id], model_storage_structure.block_storage_bytes_list[block_id]) 