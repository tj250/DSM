from multiprocessing import Process
from mcp_regression_build_model import mcp_reg_build_model
from mcp_regression_predict import mcp_reg_predict
from data_explore.data_explore import mcp_explore_data
from task_manage import start_background_task
from utility import print_with_color

'''
建模相关的tool
'''

def build_model():
    try:
        mcp_reg_build_model.run(transport="sse")
        print("执行结束")
    except Exception as e:
        print(f"❌ 线程启动失败: {str(e)}")

'''
预测相关的tool
'''
def prediction():
    try:
        mcp_reg_predict.run(transport="sse")
        print("执行结束")
    except Exception as e:
        print(f"❌ 线程启动失败: {str(e)}")

'''
数据探索相关的tool
'''
def explore_data():
    try:
        mcp_explore_data.run(transport="sse")
        print("执行结束")
    except Exception as e:
        print(f"❌ 线程启动失败: {str(e)}")


if __name__ == "__main__":
    # abc = asyncio.run( get_feature_unique_values("D:\PycharmProjects\GeoMinerAgents\pHEnv2.csv", "土地利用"))
    start_background_task() # 开始后台进程执行任务
    # 创建两个工作线程
    print_with_color("正在创建多个进程...")
    process_prediction = Process(target=prediction)
    process_explore_data = Process(target=explore_data)
    process_build_model = Process(target=build_model)  # 回归建模

    print_with_color("进程已启动")
    process_prediction.start()
    process_explore_data.start()
    process_build_model.start()

    # 等待线程结束
    print_with_color("等待进程执行完成...")
    process_prediction.join()
    process_explore_data.join()
    process_build_model.join()

    # mcp_reg_build_model.run(transport="sse")
    # mcp_reg_predict.run(transport="sse")

    # abc = asyncio.run(get_model_metrics('042c78ee-23f6-11f0-9fc4-0433c205c543'))

    # asyncio.run(accept_task_for_build_XGBR(
    #     call_id="g--------------------------------------",
    #     structure_data_file="sadfg",
    #     predict_variable="gfj",
    #     interpretation_variables="年均温,年降水,蒸散量,母岩,DEM,地形部位,TWI,LAI,植被类型,土地利用",
    #     categorical_variables='母岩,地形部位,植被类型,土地利用'
    # ))

    # asyncio.run(accept_raster_predict_task(
    #     call_id="g--------------------------------------",
    #     model_id="sadfg",
    #     template_file="gfj",
    #     interpretation_variables_file_storage_directory="年均温,年降水,蒸散量,母岩,DEM,地形部位,TWI,LAI,植被类型,土地利用",
    # ))

    # abc = asyncio.run(get_unique_value_of_structured_data_column("D:/PycharmProjects/GeoMinerAgents/pHEnv2.csv",'母岩'))



