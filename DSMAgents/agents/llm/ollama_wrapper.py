# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)  # 输出请求详情
from langchain_ollama import ChatOllama

'''
使用ollama方式调用QWQ32B的类
'''


class OllamaWrapper:

    def __init__(self, model_url, model_name):
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0,  # Ollama 中已安装的模型名称
            api_key="ollama",  # 任意非空字符串（Ollama 无需真实密钥）
            base_url=model_url  # Ollama API 地址
        )

    '''
    构建带结构化输出的llm runnable,默认包含原始消息
    '''
    def with_structured_output(self, pydantic_class: BaseModel):
        return self.llm.with_structured_output(pydantic_class, include_raw=True)

    async def execute_chat(self, prompt):
        chat_response = self.llm.invoke(prompt)
        return chat_response
