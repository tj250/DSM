from typing import List, Dict, Union
from pydantic import BaseModel
import json
from langchain_core.messages import HumanMessage,AIMessage
# 定义消息基类
class Message(BaseModel):
    type: str  # human/ai/tool
    content: Union[str, Dict]
    metadata: Dict = {}

# 格式化处理器
class TagFormatter:
    @staticmethod
    def to_json(messages: List[Message]) -> str:
        """转换为标准JSON格式"""
        return json.dumps(
            [msg.dict() for msg in messages],
            indent=2,
            ensure_ascii=False
        )

    @staticmethod
    def to_markdown(messages: List[Message]) -> str:
        """生成Markdown报告"""
        report = ["## DeepSeek-R1 推理标签报告\n"]
        for i, msg in enumerate(messages, 1):
            report.append(f"### 消息 {i} ({msg.type.upper()})")
            report.append(f"&zwnj;**内容**&zwnj;: {msg.content}")
            if msg.response_metadata:
                report.append("&zwnj;**元数据**&zwnj;:")
                report.append(f"```json\n{json.dumps(msg.response_metadata, indent=2)}\n```")
            report.append("---")
        return "\n".join(report)

    @classmethod
    def format(
        cls,
        messages: List[Message],
        style: str = "markdown"
    ) -> Union[str, Dict]:
        """统一入口方法"""
        formatters = {
            "json": cls.to_json,
            "markdown": cls.to_markdown
        }
        return formatters[style](messages)
