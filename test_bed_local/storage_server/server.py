import grpc
from concurrent import futures
import os
from  test_bed_local.proto.storage_pb2 import *
from  test_bed_local.proto.storage_pb2_grpc import *

class StorageService(StorageServicer):
    def __init__(self, base_directory, chunk_size=2 * 1024 * 1024):
        self.base_directory = base_directory
        self.chunk_size = chunk_size  # 每块大小为 4MB

    def GetFiles(self, request, context):
        model_directory = os.path.join(self.base_directory, request.model_name)

        if not os.path.exists(model_directory):
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details('Model not found')
            return
        
        print('model_directory',request.model_name)

        for filename in os.listdir(model_directory):
            file_path = os.path.join(model_directory, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    while chunk := f.read(self.chunk_size):
                        yield FileChunk(
                            filename=filename,
                            content=chunk,
                            end_of_file=False,
                        )
                # 文件结束标志
                yield FileChunk(
                    filename=filename,
                    content=b'',
                    end_of_file=True,
                )

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_StorageServicer_to_server(
        StorageService(base_directory="../serve/model_info/models/llama"), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
