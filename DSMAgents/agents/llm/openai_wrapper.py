import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG)  # 输出请求详情
from langchain_openai import ChatOpenAI

'''
使用ollama方式调用QWQ32B的类
'''


class OpenAIWrapper:

    def __init__(self, model_url, model_name, api_key):
        self.llm = ChatOpenAI(
            base_url=model_url,
            api_key=api_key,  # 替换为你的实际API密钥
            model=model_name,  # 可替换为其他OpenRouter支持的模型
            temperature=0.7,
            max_tokens=2000
        )
    '''
    构建带结构化输出的llm runnable,默认包含原始消息
    '''
    def with_structured_output(self, pydantic_class: BaseModel):
        return self.llm.with_structured_output(pydantic_class, include_raw=True)

    async def execute_chat(self, prompt):
        chat_response = self.llm.invoke(prompt)
        return chat_response
