from tools_data_access import RegressionModelInfo, TaskDataAccess
from mcp.server.fastmcp import FastMCP
from utility import print_caller_parameters

mcp_reg_build_model = FastMCP("RegressionBuildModel", port=8002)

@mcp_reg_build_model.tool()
async def accept_task_for_build_XGBR(call_id: str, structure_data_file: str, predict_variable: str,
                        interpretation_variables:str, categorical_variables: str) -> tuple[str,str]:
    """
    接收建立XGBoost回归模型的输入参数并记录于数据库

    Args:
        call_id (str):本次回归建模任务的调用ID。
        structure_data_file (str): 结构化数据文件。
        predict_variable (str): 预测变量。
        interpretation_variables (str): 解释变量。
        categorical_variables (str): 类别型变量。

    Returns:
        tuple[str,str]: 所采用的模型以及所开启的建模任务ID。
    """
    print_caller_parameters()
    input_params = '|'.join([structure_data_file,predict_variable,interpretation_variables,categorical_variables])
    model_name = 'XGBoost回归'
    build_model_task_id = TaskDataAccess.add_build_model_info(call_id, model_name, input_params)
    return model_name, build_model_task_id


@mcp_reg_build_model.tool()
async def accept_task_for_build_XGBRFR(call_id: str, structure_data_file: str, predict_variable: str,
                                       interpretation_variables:str, categorical_variables: str) -> tuple[str,str]:
    """
    接收建立XGBoost随机森林回归模型的输入参数并记录于数据库

    Args:
        call_id (str):本次回归建模任务的调用ID。
        structure_data_file (str): 结构化数据文件。
        predict_variable (str): 预测变量。
        interpretation_variables (str): 解释变量。
        categorical_variables (str): 类别型变量。

    Returns:
        tuple[str,str]: 所采用的模型以及所开启的建模任务ID。
    """
    print_caller_parameters()
    input_params = '|'.join([structure_data_file,predict_variable,interpretation_variables,categorical_variables])
    model_name = 'XGBoost随机森林回归'
    model_id = TaskDataAccess.add_build_model_info(call_id, model_name, input_params)
    return model_name, model_id


@mcp_reg_build_model.tool()
async def accept_task_for_build_SVR(call_id: str, structure_data_file: str, predict_variable: str,
                                    interpretation_variables:str, categorical_variables: str) -> tuple[str,str]:
    """
    接收建立支持向量机回归模型的输入参数并记录于数据库

    Args:
        call_id (str):本次回归建模任务的调用ID。
        structure_data_file (str): 结构化数据文件。
        predict_variable (str): 预测变量。
        interpretation_variables (str): 解释变量。
        categorical_variables (str): 类别型变量。

    Returns:
        tuple[str,str]: 所采用的模型以及所开启的建模任务ID。
    """
    print_caller_parameters()
    input_params = '|'.join([structure_data_file,predict_variable,interpretation_variables,categorical_variables])
    model_name = '支持向量机回归'
    model_id = TaskDataAccess.add_build_model_info(call_id, model_name, input_params)
    return model_name, model_id


@mcp_reg_build_model.tool()
async def task_is_finished(call_id: str) -> bool:
    """
    获取一次回归建模任务相关的建模过程是否结束。

    Args:
        call_id (str): 本次回归建模任务的调用ID。

    Returns:
        bool: 与该次回归建模任任务相关的建模过程是否已经结束
    """
    print_caller_parameters()
    return TaskDataAccess.build_model_task_is_finished(call_id)

@mcp_reg_build_model.tool()
async def get_model_metrics(model_id: str) -> RegressionModelInfo | None:
    """
    获取回归模型的指标信息。

    Args:
        model_id (str): 模型的ID。

    Returns:
        RegressionModelInfo: 获取成功时则为模型的指标信息，否则为None。
    """
    print_caller_parameters()
    successful, metrics = TaskDataAccess.get_model_info(model_id)
    if successful:
        return metrics
    else:
        return None
