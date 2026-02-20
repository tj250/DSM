# 定义定时任务
import threading
import time
from tools_data_access import TaskDataAccess,RegressionModelInfo, PredictionInfo
from build_regression_model import build_XGBR,build_XGBRFR,build_SVR
from regression_predict import create_raster
from utility import print_with_color
from colorama import Fore

mutex = threading.Lock()

def timed_task():
    # 获取锁（阻塞直到成功）
    get_mutex = mutex.acquire(blocking=False)
    if get_mutex: # 如果获取到了锁
        try:
            # print_with_color(f"{threading.current_thread().name} 获取锁，执行任务", color=Fore.YELLOW)
            next_task = TaskDataAccess.get_next_task()
            if next_task is not None:
                if type(next_task) is RegressionModelInfo:
                    input_params = next_task.input_params.split('|')
                    if next_task.model_name == 'XGBoost回归':
                        build_XGBR(next_task.model_id, input_params[0], input_params[1], input_params[3].split(','))
                    elif next_task.model_name == 'XGBoost随机森林回归':
                        build_XGBRFR(next_task.model_id, input_params[0], input_params[1], input_params[3].split(','))
                    elif next_task.model_name == '支持向量机回归':
                        build_SVR(next_task.model_id, input_params[0], input_params[1], input_params[3].split(','))
                    print_with_color(f"处理如下建模任务成功:{next_task.model_id} ", color=Fore.YELLOW)
                elif type(next_task) is PredictionInfo:
                    input_params = next_task.input_params.split('|')
                    if next_task.prediction_type == 1: # 栅格预测
                        _,model_info = TaskDataAccess.get_model_info(next_task.model_id)
                        build_model_params = model_info.input_params.split('|')
                        if create_raster(next_task.prediction_id,next_task.call_id,next_task.model_id, input_params[0], input_params[1],
                                         build_model_params[2].split(',')):
                            TaskDataAccess.set_prediction_task_finished(next_task.prediction_id)  # 保存到数据库,以供后续查询

                    print_with_color(f"处理如下预测任务成功:{next_task.prediction_id} ", color=Fore.YELLOW)

        finally:
            # 释放锁
            mutex.release()
            # print_with_color(f"{threading.current_thread().name} 释放锁", color=Fore.YELLOW)
    else:
        print_with_color(f"{threading.current_thread().name} 锁被其他处理任务占用，无法获取锁，跳出。", color=Fore.YELLOW)

'''
开启定时任务，遍历数据库中记录的还没有执行的任务，然后执行
'''
def start_background_task():
    timer = RepeatingTimer(interval=5, func=timed_task)
    timer.start()  # 启动定时器


class RepeatingTimer:
    def __init__(self, interval, func):
        self.interval = interval  # 间隔时间（秒）
        self.func = func          # 要执行的任务函数
        self.timer = None         # 定时器对象
        self.is_running = False   # 运行状态标志

    def _run(self):
        self.is_running = False
        self.start()              # 递归启动新定时器
        self.func()               # 执行任务

    def start(self):
        if not self.is_running:
            self.timer = threading.Timer(self.interval, self._run)
            self.timer.start()
            self.is_running = True

    def stop(self):
        if self.timer:
            self.timer.cancel()   # 终止定时器
        self.is_running = False