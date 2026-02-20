from mcp.server.fastmcp import FastMCP
from tools_data_access import RegressionModelInfo, TaskDataAccess
from mcp_regression_build_model import mcp_reg_build_model
from utility import print_caller_parameters

mcp_reg_predict = FastMCP("RegressionPredict", port=8003)


@mcp_reg_predict.tool()
async def accept_raster_predict_task(call_id: str, model_id:str, template_file:str,
                                     interpretation_variables_file_storage_directory: str, ) -> bool:
    """
    接收有关栅格数据的回归预测任务，将任务相关参数记录于数据库

    Args:
        call_id (str):本次回归预测任务的调用ID。
        model_id (str):回归所采用的模型的ID，即model_id。
        template_file (str): 解释变量对应的栅格数据文件列表。
        interpretation_variables_file_storage_directory (str): 解释变量对应的栅格数据文件所存储的目录。

    Returns:
        bool: 是否成功记录了该次预测任务。
    """
    print_caller_parameters()
    input_params = '|'.join([template_file, interpretation_variables_file_storage_directory])
    prediction_id = TaskDataAccess.add_prediction_task_info(call_id, model_id, 1, input_params) # 1:表示栅格数据预测
    return True

@mcp_reg_predict.tool()
async def prediction_task_is_finished(call_id: str) -> bool:
    """
    获取一次回归预测任务是否结束。

    Args:
        call_id (str): 本次回归预测任务的调用ID。

    Returns:
        bool: 指定调用ID相关的回归预测任务是否均已结束。
    """
    print_caller_parameters()
    return TaskDataAccess.prediction_task_is_finished(call_id)