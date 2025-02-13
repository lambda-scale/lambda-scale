from enum import Enum


class TransferStrategyEnum(Enum):
    BinomialPipeline = 1
    Local = 2
    NcclBroadcast = 3
    FaaSnet = 4
    Sllm = 5

class ExecuteStrategyEnum(Enum):
    DynamicPP = 1
    LLMDynamicPP = 2
    No = 3