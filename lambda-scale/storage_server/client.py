import time
import grpc
import os
from  test_bed_local.proto.storage_pb2 import *
from  test_bed_local.proto.storage_pb2_grpc import *

class StoreClient:
    def __init__(self, server_address="127.0.0.1:50051"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = StorageStub(self.channel)

    def get_files(self, model_name, save_directory):
        request = GetFilesRequest(model_name=model_name)
        try:
            responses = self.stub.GetFiles(request)
            current_file = None
            file_path = None

            for response in responses:
                if current_file != response.filename:
                    if current_file:
                        print(f"Finished downloading {current_file}")
                    current_file = response.filename
                    file_path = os.path.join(save_directory, current_file)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                with open(file_path, 'ab') as f:
                    f.write(response.content)

                if response.end_of_file:
                    print(f"Completed: {current_file}")

        except grpc.RpcError as e:
            print(f"RPC failed: {e}")

