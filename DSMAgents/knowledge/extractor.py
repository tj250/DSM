import textwrap
from langextract import data as lx_data
import langextract as lx
from env_covariates_pydantic import FaultInfo

def extract():
    prompt_desc = textwrap.dedent("""  
    从工业设备故障报告中，按出现顺序抽取出故障信息。  
    请精确使用原文中的文本进行抽取，不要转述或概括。  
    为每个实体提供有意义的属性。""")

    example_text = "记录编号：20250910-01。巡检发现，P-101离心泵出口压力下降20%，伴随异常振动。初步判断为叶轮堵塞。处理措施：停机并清理叶轮异物。"
    example_extractions = [
        lx_data.Extraction(
            extraction_class="FaultInfo",
            extraction_text="P-101离心泵出口压力下降20%，伴随异常振动。初步判断为叶轮堵塞。处理措施：停机并清理叶轮异物。",
            attributes={
                "equipment_name": "P-101离心泵",
                "phenomenon": "出口压力下降20%，伴随异常振动",
                "cause": "叶轮堵塞",
                "solution": "停机并清理叶轮异物"
            }
        )
    ]
    example = lx_data.ExampleData(text=example_text, extractions=example_extractions)

    # 输入待处理的新文本
    input_text = "2025年9月11日，当班操作员报告，V-201反应釜的机械密封出现泄漏，现场有明显物料滴落。经检查，确认为法兰连接螺栓松动导致。已派维修工紧固所有螺栓，泄漏停止。"

    # 运行抽取（需配置本地Ollama模型）
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt_desc,
        examples=[example],
        # format_type=[FaultInfo],  # 指定目标Schema
        # 本地模型
        model_id="qwen3:14b",  # or any Ollama model
        model_url="http://192.168.1.71:11434",
        fence_output=False,
        use_schema_constraints=False

    )

    for extraction in result.extractions:
        if extraction.extraction_class == 'FaultInfo':
            # 将抽取结果转为我们定义的FaultInfo对象
            fault_info = FaultInfo(**extraction.attributes)
            print(f"设备: {fault_info.equipment_name}")
            print(f"现象: {fault_info.phenomenon}")
            print(f"原因: {fault_info.cause}")
            print(f"措施: {fault_info.solution}\n")

if __name__ == "__main__":
    extract()