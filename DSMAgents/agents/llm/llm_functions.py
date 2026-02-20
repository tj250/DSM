from loguru import logger
import json_repair, xmltodict
from xml.parsers.expat import ExpatError
from langchain_core.utils.json import parse_json_markdown
from agents.llm.vllm_wapper import VLLMWapper
from agents.llm.ollama_wrapper import OllamaWrapper
from agents.llm.openai_wrapper import OpenAIWrapper
from pydantic import BaseModel  # 数据验证和字段定义工具

llm_choice = "Ollama"  # "Ollama" or "vLLM" or "OpenAI"

# Ollama模型配置参数
ollama_model_url = "http://localhost:11434"
ollama_model_name = 'qwen3:14b'
# 192.168.1.71：'qwen3:14b' 'qwen3:30b' 'qwen3:32b'
# 192.168.1.95：'qwen3:14b'

# OpenAI兼容的配置参数，此处使用openrouter提供的接口
# openai_model_name = "deepseek/deepseek-chat-v3.1:free"
openai_model_name = "qwen/qwen3-14b:free"
# openai_model_name = "qwen/qwen3-235b-a22b:free"
openai_base_url = "https://openrouter.ai/api/v1"
openai_api_key = "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"

# vLLM兼容的配置参数
vLLM_model_url = "http://192.168.1.71:8000/v1"

'''
调用LLM以产生输出
'''


async def call_llm(prompt: list, response_format: str | None = None, ):
    if llm_choice == 'Ollama':
        model_class = OllamaWrapper(ollama_model_url, ollama_model_name)
    elif llm_choice == 'OpenAI':
        model_class = OpenAIWrapper(openai_base_url, openai_model_name, openai_api_key)
    else:
        model_class = VLLMWapper(vLLM_model_url)
    try:
        if response_format == "json":
            response = await model_class.execute_chat(prompt)
            return parse_json_markdown(response, parser=json_repair.loads)
        elif response_format == "dict":
            try_xml_parse_times = 0
            while try_xml_parse_times < 3:
                response = await model_class.execute_chat(prompt)
                print(response.content)
                # 解析LLM给出的XML格式内容
                try:
                    response_dict = xmltodict.parse("<top>{}</top>".format(response.content))
                    try_xml_parse_times = 10  # 跳出解析
                except ExpatError as e:  # 由于LLM能力所限，会生成语法错误的XML，如标签未闭合/嵌套错误等
                    print(response.content)
                    try_xml_parse_times += 1
                    response_dict = {"top": {"error": str(e)}}
            return response_dict["top"], response
    except Exception as e:
        print("调用模型发生错误")
        logger.error(f"Error in calling model: {e}")


'''
以严格的pydantic模式输出LLM的调用结果。此方法内部为同步执行，需要等待LLM一次性给出最终的输出结果
'''


def call_llm_with_pydantic_output(prompt: list, pydantic_class: BaseModel):
    if llm_choice == 'Ollama':
        model_class = OllamaWrapper(ollama_model_url, ollama_model_name)  # 'qwq:latest' 'deepseek-r1:14b'
    elif llm_choice == 'OpenAI':
        model_class = OpenAIWrapper(openai_base_url, openai_model_name, openai_api_key)
    else:
        model_class = VLLMWapper(vLLM_model_url)
    try:
        structured_llm = model_class.with_structured_output(pydantic_class)
        return structured_llm.invoke(prompt)
    except Exception as e:
        print("调用模型发生错误")
        logger.error(f"Error in calling model: {e}")


'''
获得当前配置参数所决定的LLM，如何需要LLM的异步输出流时，调用此方法。
'''


def get_llm_with_pydantic_output(pydantic_class: BaseModel):
    return get_llm().with_structured_output(pydantic_class, include_raw=True)


'''
获得当前配置参数所决定的LLM
'''


def get_llm():
    if llm_choice == 'Ollama':
        model_class = OllamaWrapper(ollama_model_url, ollama_model_name)
    elif llm_choice == 'OpenAI':
        model_class = OpenAIWrapper(openai_base_url, openai_model_name, openai_api_key)
    else:
        model_class = VLLMWapper(vLLM_model_url)
    return model_class.llm
