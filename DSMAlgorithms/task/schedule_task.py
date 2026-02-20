# 定义定时任务
import threading
from utility import print_with_color
from colorama import Fore
from db_access.task_dealer import TaskDataAccess
from task.build_model import execute_algorithm, stacking_models
import algorithms_config
mutex = threading.Lock()

exception_occurred = False  # 控制是否在发生异常后，还要继续循环处理

def timed_task():
    global exception_occurred
    if exception_occurred and not algorithms_config.DOCKER_MODE:  # 非Docker模式是，如果发生异常后，则不再尝试
        print('异常发生')
        return
    # 获取锁（阻塞直到成功）
    get_mutex = mutex.acquire(blocking=False)
    if get_mutex: # 如果获取到了锁
        try:
            # print_with_color(f"{threading.current_thread().name} 获取锁，执行任务", color=Fore.YELLOW)
            next_task_info, task_type = TaskDataAccess.get_next_task()
            if len(next_task_info) != 0:
                if task_type == 1: # 单一算法建模
                    print_with_color(f"开始处理建模任务:{next_task_info['algorithms_id']}|\
                    {next_task_info['algorithms_name']}", color=Fore.YELLOW)
                    successful = execute_algorithm(next_task_info['algorithms_id'],
                                      next_task_info['algorithms_name'],
                                      next_task_info['prediction_variable'],
                                      next_task_info['file_name'],
                                      next_task_info['covariates_directory'],
                                      next_task_info['categorical_vars'],
                                      next_task_info['left_cols1'],
                                      next_task_info['left_cols2'],
                                      next_task_info['left_cols3'],
                                      next_task_info['data_distribution_type'],
                                      )
                    if successful:
                        print_with_color(f"处理如下建模任务成功:{next_task_info['algorithms_id']} ", color=Fore.GREEN)
                    else:
                        print_with_color(f"处理如下建模任务失败:{next_task_info['algorithms_id']} ", color=Fore.RED)
                elif task_type == 2: # 堆叠算法
                    print_with_color(f"开始处理建模任务(堆叠):{next_task_info['stacking_algorithms_id']}|\
                                        {'|'.join(next_task_info['algorithms_name'])}",
                                     color=Fore.YELLOW)
                    successful = stacking_models(next_task_info['stacking_algorithms_id'],
                                      next_task_info['algorithms_name'],
                                      next_task_info['prediction_variable'],
                                      next_task_info['file_name'],
                                      next_task_info['covariates_directory'],
                                      next_task_info['categorical_vars'],
                                      next_task_info['left_cols1'],
                                      next_task_info['left_cols2'],
                                      next_task_info['left_cols3'],
                                      next_task_info['data_distribution_type'],)
                    if successful:
                        print_with_color(f"处理如下模型堆叠任务成功:{next_task_info['stacking_algorithms_id']} ", color=Fore.GREEN)
                    else:
                        print_with_color(f"处理如下模型堆叠任务失败:{next_task_info['stacking_algorithms_id']} ",
                                         color=Fore.RED)
        except Exception as ex:
            exception_occurred = True
            raise ex
            # print(f"{ex}")
        finally:
            # 释放锁
            mutex.release()
            # print_with_color(f"{threading.current_thread().name} 释放锁", color=Fore.YELLOW)
    else:
        pass
        # print_with_color(f"{threading.current_thread().name} 锁被其他处理任务占用，无法获取锁，跳出。", color=Fore.YELLOW)

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