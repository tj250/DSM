from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class FaultInfo:
    """描述一次设备故障的详细信息。"""
    equipment_name: Optional[str] = field(default=None, metadata={"description": "发生故障的设备名称，例如 'P-101 离心泵'"})
    phenomenon: Optional[str] = field(default=None, metadata={"description": "观察到的具体故障现象，例如 '出口压力下降20%'"})
    cause: Optional[str] = field(default=None, metadata={"description": "分析得出的根本原因，例如 '密封环磨损'"})
    solution: Optional[str] = field(default=None, metadata={"description": "采取的维修或解决方案，例如 '更换O型密封环'"})

@dataclass
class FaultReport:
    """从一份报告中抽取的故障信息列表。"""
    faults: List[FaultInfo]