from test_bed_local.serve.utils.utils import read_evaluation_parameters


params = read_evaluation_parameters()

print(f"Total nodes: {params.get('total_node_num')}")
print(f"Model name: {params.get('model_name')}")
print(f"Model ID: {params.get('model_id')}")
print(f"Is SSD exist: {params.get('is_ssd_exist')}")
print(f"Is CPU exist: {params.get('is_cpu_exist')}")
print(f"Is remote storage: {params.get('is_remote_storage')}")