def build_package_proto(root: str, proto_file: str) -> None:
    from grpc_tools import protoc

    command = [
        "grpc_tools.protoc",
        "-I",
        "./",
        f"--python_out={root}",
        f"--grpc_python_out={root}",
    ] + [proto_file]
    if protoc.main(command) != 0:
        raise RuntimeError("error: {} failed".format(command))
    
build_package_proto(root = "/jiachaobo/test/gpu-fast-scaling/test_bed_local/proto",
                    proto_file="storage.proto")
        
