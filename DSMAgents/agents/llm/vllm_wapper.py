# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.adapters.openai import convert_openai_messages


class LocalChatOpenAI(ChatOpenAI):
    @property
    def _llm_params(self):
        params = super()._llm_params
        del params["api_key"]  # 移除密钥校验
        return params


'''
使用vLLM方式调用QWQ32B的类
'''


class VLLMWapper:

    def __init__(self, model_url):
        # self.llm = OpenAI(
        #     api_key="EMPTY",  # vLLM无需API密钥
        #     base_url=model_url,
        #     model_name="qwq-32b"
        # )
        # # 初始化 OpenAI 兼容接口
        # llm = OpenAI(
        #     model_name="qwq-32b",  # 需与 API 服务端定义的模型标识一致 ‌:ml-citation{ref="3,7" data="citationList"}
        #     api_key="EMPTY",
        #     base_url=os.getenv("OPENAI_BASE_URL"),  # 指定自定义端点 ‌:ml-citation{ref="2,3" data="citationList"}
        #     streaming=True,  # 启用流式输出
        #     callbacks=[StreamingStdOutCallbackHandler()]  # 实时逐词输出 ‌:ml-citation{ref="1,7" data="citationList"}
        # )
        # 方式3-配置指向本地 vLLM 服务
        self.llm = LocalChatOpenAI(
            base_url=model_url,  # vLLM 服务地址
            model="qwq-32b",  # 需与vLLM启动参数--model一致
            # api_key="EMPTY"  绕过密钥验证
        )

    async def execute_chat(self, prompt):
        lc_messages = convert_openai_messages(prompt)
        # chat_response = await self.llm.chat.completions.create(
        #     messages=lc_messages,
        #     temperature=0.6,
        #     top_p=0.95,
        #     max_tokens=1024000,
        #     extra_body={"repetition_penalty": 1.05,},
        # )

        # 方式2或3-调用模型
        chat_response = self.llm.invoke(lc_messages)
        return chat_response
