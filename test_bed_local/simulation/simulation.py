from collections import deque

def parse_trace(file_path):
    data = []
    first_timestamp = None

    with open(file_path, 'r') as f:
        header = f.readline()  
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) != 3:
                continue

            timestamp_str, context_tokens_str, generated_tokens_str = parts

            if '.' in timestamp_str:
                main_part, micro_part = timestamp_str.split('.')
                micro_part = micro_part[:3]  
                timestamp_str = f"{main_part}.{micro_part}"
            else:
                timestamp_str += ".000"  

            try:
                date_part, time_part = timestamp_str.split(' ')
                year, month, day = map(int, date_part.split('-'))
                hour, minute, second = map(float, time_part.split(':'))
                
                total_seconds = hour * 3600 + minute * 60 + second
            except ValueError:
                print(f"时间戳解析错误: {timestamp_str}")
                continue  

            if first_timestamp is None:
                first_timestamp = total_seconds
                delta = 0.0
            else:
                delta = total_seconds - first_timestamp
                delta = round(delta, 3)  

            try:
                context_tokens = int(context_tokens_str)
                generated_tokens = int(generated_tokens_str)
            except ValueError:
                print(f"data error: {line}")
                continue 

            data.append((delta, context_tokens, generated_tokens))

    return data

class ModelServerSimulation:
    def __init__(self, gpu_keep_alive_time,keep_alive_time, request_trace, cluster_size=100, ssd_to_memory_time=20, memory_to_gpu_time=5, serve_time=3, memory_cost_per_model=140):
        self.keep_alive_time = keep_alive_time  # Time to keep model in memory cache
        self.gpu_keep_alive_time = gpu_keep_alive_time
        self.ssd_to_memory_time = ssd_to_memory_time
        self.memory_to_gpu_time = memory_to_gpu_time
        self.serve_time = serve_time
        self.request_trace = deque([(i, req[0], req[1]) for i, req in enumerate(request_trace)])  # A queue of (request_id, timestamp, model) for incoming requests

        self.cluster_size = cluster_size  # Number of initial nodes in the cluster
        self.nodes = [
            {"gpu_model": None, "memory_cache": {},"gpu_model_last_access":None, "current_task": None} for _ in range(cluster_size)
        ]  # Each node tracks its GPU model, memory cache, and current task

        self.memory_cost_per_model = memory_cost_per_model  # Memory cost for each cached model
        self.total_memory_time_cost = 0  # Tracks the total memory cost (occupied memory * time)

        self.total_requests = 0
        self.cold_starts = 0  # Requests requiring SSD load
        self.over_provision_time = 0  # Total time memory cache held unused models
        self.over_provision_ratio = 0  # Over-provision ratio across time
        self.current_time = 0

        self.task_queue = deque()  # Queue of tasks waiting to be executed

    def simulate(self):
        while self.request_trace or self.task_queue or any(node["current_task"] for node in self.nodes):
            # Process incoming requests at the current time
            while self.request_trace and self.request_trace[0][1] == self.current_time:
                request_id, _, model = self.request_trace.popleft()
                self.total_requests += 1
                self.task_queue.append((request_id, model, "request"))

            # Process task queue on nodes
            for node in self.nodes:
                if node["current_task"]:
                    task = node["current_task"]
                    task["remaining_time"] -= 1
                    if task["remaining_time"] <= 0:
                        self._complete_task(node)
                elif self.task_queue:
                    request_id, model, task_type = self.task_queue.popleft()
                    self._assign_task(node, request_id, model, task_type)

            self._cleanup_gpu_cache()

            # Cleanup memory cache for unused models
            self._cleanup_memory_cache()

            # Increment memory cost at the current time
            self._track_memory_cost()

            # Calculate over-provision for the current time step
            self._track_over_provision()



            # Increment time
            self.current_time += 1

        # Return results
        return {
            "cold_start_rate": self.cold_starts / self.total_requests,
            "over_provision_rate": self.over_provision_time / self.current_time,
            "total_memory_time_cost": self.total_memory_time_cost,
        }

    def _assign_task(self, node, request_id, model, task_type):
        """Assign a task to a node if possible."""
        if task_type == "request":
            if node["gpu_model"] == model:
                node["gpu_model_last_access"] = None
                # print('Serve directly from GPU')
                # Serve directly from GPU
                node["current_task"] = {"type": "serve", "remaining_time": self.serve_time, "request_id": request_id}
            elif model in node["memory_cache"]:
                # Load from memory to GPU
                # print('Load from memory to GPU')
                node["current_task"] = {"type": "load_memory_to_gpu", "model": model, "remaining_time": self.memory_to_gpu_time + self.serve_time, "request_id": request_id}
            elif not node["gpu_model"]:
                # print('Load from SSD to memory, then to GPU')
                # Load from SSD to memory, then to GPU
                node["current_task"] = {"type": "load_ssd_to_gpu", "model": model, "remaining_time": self.ssd_to_memory_time + self.memory_to_gpu_time + self.serve_time, "request_id": request_id}

    def _complete_task(self, node):
        """Complete the current task on a node."""
        task = node["current_task"]
        if task["type"] == "serve":
            node["gpu_model_last_access"] = self.current_time
        elif task["type"] == "load_memory_to_gpu":
            node["gpu_model"] = task["model"]
            node["gpu_model_last_access"] = self.current_time
        elif task["type"] == "load_ssd_to_gpu":
            node["gpu_model"] = task["model"]
            node["gpu_model_last_access"] = self.current_time
            node["memory_cache"][task["model"]] = self.current_time
            self.cold_starts += 1

        node["current_task"] = None 

    def _cleanup_memory_cache(self):
        """Remove models from memory cache if they exceed keep-alive time."""
        for node in self.nodes:
            to_remove = []
            for model, last_access in node.get("memory_cache", {}).items():
                # print(self.current_time,last_access,self.keep_alive_time)
                if self.current_time - last_access > self.keep_alive_time:
                    to_remove.append(model)

            for model in to_remove:
                node["memory_cache"].pop(model)
    def _cleanup_gpu_cache(self):
        """Remove models from GPU cache if they exceed keep-alive time."""
        for node in self.nodes:
            if node["gpu_model_last_access"] and self.current_time - node["gpu_model_last_access"] > self.gpu_keep_alive_time:
                node["gpu_model"] = None

    def _track_memory_cost(self):
        """Track the total memory cost (occupied memory * time) at the current time."""
        current_memory_usage = 0
        for node in self.nodes:
            current_memory_usage += len(node["memory_cache"]) * self.memory_cost_per_model
        self.total_memory_time_cost += current_memory_usage

    def _track_over_provision(self):
        """Track over-provisioned memory at the current time."""
        active_nodes = 0
        total_cached_models = 0

        for node in self.nodes:
            if node["current_task"] and node["current_task"]["type"] == "serve":  # Only count serve tasks as active
                active_nodes += 1
            total_cached_models += len(node["memory_cache"])

        # Over-provisioned memory is total cached models minus active nodes
        over_provisioned_memory = total_cached_models - active_nodes
        if active_nodes > 0:  # Avoid division by zero
            over_provision_ratio = over_provisioned_memory / active_nodes
            self.over_provision_ratio += over_provision_ratio  # Accumulate ratio over time
        if over_provisioned_memory > 0:
            self.over_provision_time += 1  # Count this time step as over-provisioned

# Example usage with a random request trace
if __name__ == "__main__":
    import random
    random.seed(42)
    # Generate a random request trace with timestamps and random model IDs

    num_requests = 2
    max_gap = 10

    request_trace = []
    current_time = 0

    for i in range(num_requests):
        if i % 2 == 0: 
            gap = random.randint(0, 1)  
        else:  
            gap = random.randint(2, max_gap)  
        current_time += gap
        request_trace.append((current_time, 0))

    # print(request_trace)

    # request_trace = [(0, 0), (2, 0),(50,0),(51,0),(52,0)]

    trace_file = 'trace.txt'
    parsed_data = parse_trace(trace_file)

    # print(parsed_data[1500:2500])

    data = parsed_data[1500:2500]

    request_trace = []

    for info in data:
        request_trace.append((int(info[0]*10)-3097,0))

    # print(request_trace)

    for keep_alive in [0,10,20,30,60,120]:
        simulation = ModelServerSimulation(gpu_keep_alive_time = 0,
                                           keep_alive_time=keep_alive, 
                                           request_trace=request_trace)
        results = simulation.simulate()
        print(f"Keep-alive: {keep_alive}s -> Cold start rate: {results['cold_start_rate']:.2%}, Over-provision rate: {results['over_provision_rate']:.2%}, Total memory time cost: {results['total_memory_time_cost']} GB.s")
