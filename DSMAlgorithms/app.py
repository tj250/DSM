from flask import Flask, jsonify
from task.schedule_task import start_background_task
from DSMAlgorithms.base.base_data_structure import DataDistributionType

app = Flask(__name__)


@app.route("/api/hello")
def hello():
    return jsonify({"message": "Hello from Docker!"})


'''
执行具体的建模算法
model_id:模型的ID，对应于build_model表中的model_id字段
algorithm：算法名称
'''


@app.route("/call_model", methods=["POST"])
def call_model():
    # 接收JSON格式参数
    # request_data = request.get_json()  # 自动解析请求体为字典
    # algorithm_id = request_data["algorithm_id"]
    # algorithm_name = request_data["algorithm"]
    # prediction_variable = request_data["prediction_variable"]
    # file_name = request_data["file_name"]
    # covariates_dir_name = request_data["covariates_directory"]
    # categorical_vars = request_data["categorical_vars"]
    # data_distribution_type = DataDistributionType(request_data["data_distribution_type"])
    # left_cols = request_data["left_cols"]).split(',')
    # print(f"收到建模请求：{request_data}")
    # # 开启线程执行，以便尽快返回http调用的结果
    # t = threading.Thread(target=execute_algorithm, kwargs={"algorithm_id": algorithm_id,
    #                                                        "algorithms_type": AlgorithmsType(algorithm_name),
    #                                                        "prediction_variable": prediction_variable,
    #                                                        "file_name": file_name,
    #                                                        "covariates_dir_name": covariates_dir_name,
    #                                                        "categorical_vars": categorical_vars,
    #                                                        "left_cols": left_cols,
    #                                                        "data_distribution_type": data_distribution_type})
    # t.start()
    return {}


@app.route("/stacking", methods=["POST"])
def stacking():
    # 接收JSON格式参数
    # request_data = request.get_json()  # 自动解析请求体为字典
    # stacking_algorithms_id = request_data["stacking_algorithms_id"]
    # stacking_algorithms = request_data["stacking_algorithms"]
    # prediction_variable = request_data["prediction_variable"]
    # file_name = request_data["file_name"]
    # covariates_dir_name = request_data["covariates_directory"]
    # categorical_vars = request_data["categorical_vars"]
    # left_cols = request_data["left_cols"]).split(',')
    # data_distribution_type = DataDistributionType(request_data["data_distribution_type"])
    # print(f"收到模型堆叠请求：{request_data}")
    # # 开启线程执行，以便尽快返回http调用的结果
    # t = threading.Thread(target=stacking_models, kwargs={"stacking_algorithms_id": stacking_algorithms_id,
    #                                                      "stacking_algorithms": [AlgorithmsType(name) for name in stacking_algorithms.split('|')],
    #                                                      "prediction_variable": prediction_variable,
    #                                                      "file_name": file_name,
    #                                                      "covariates_dir_name": covariates_dir_name,
    #                                                      "categorical_vars": categorical_vars,
    #                                                      "left_cols": left_cols,
    #                                                      "data_distribution_type": data_distribution_type})
    # t.start()
    return {}


if __name__ == "__main__":
    start_background_task()  # 开始后台进程执行任务
    app.run(host="0.0.0.0", port=8389)
