from setuptools import setup
setup(
        name = "test_bed_local",
        version = "0.0.1",
    
        packages=[
        "test_bed_local",
        "test_bed_local.proto",
        "test_bed_local.serve",
        "test_bed_local.tools",
        "test_bed_local.storage_server",
        "test_bed_local.serve.controller",
        "test_bed_local.serve.manager",
        "test_bed_local.serve.controller.auto_scaler",
        "test_bed_local.serve.controller.scaling_strategy",
        "test_bed_local.serve.controller.scaling_strategy.execute_strategy",
        "test_bed_local.serve.controller.scaling_strategy.transfer_strategy",
        "test_bed_local.serve.server",
        "test_bed_local.serve.utils",
        "test_bed_local.serve.communication",
        "test_bed_local.serve.model_info",
        "test_bed_local.serve.model_info.model_execute_info",
        "test_bed_local.serve.model_info.models.llama",
        "test_bed_local.serve.server.model_storage",
        "test_bed_local.test",
        "test_bed_local/third_party/fairscale/model_parallel",
        "test_bed_local/third_party/fairscale"
    ],
)