from colorama import Fore, Style
import inspect,datetime
import numpy as np
import importlib.util

def load_module_from_file(filepath):
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("dynamic_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def get_sklearn_style_class(module):
    # 获取模块中定义的类和类方法
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            methods = [m[0] for m in inspect.getmembers(obj, inspect.isfunction)]
            if '__init__' in methods and 'fit' in methods and 'predict' in methods: # 类包含三个必须的方法
                return obj
    return None

def print_with_color(output:str, color=Fore.CYAN) -> None:
    # 生成时间戳（格式：[HH:MM:SS]）
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {color}{output}{Style.RESET_ALL}")

'''
打印方法的调用参数
'''
def print_caller_parameters():
    # 获取调用者的栈帧
    caller_frame = inspect.currentframe().f_back
    # 获取调用者的参数信息
    args, varargs, varkw, locals_dict = inspect.getargvalues(caller_frame)

    print_with_color(f"函数 {caller_frame.f_code.co_name} 被调用，参数如下：")

    # 打印固定参数
    for arg_name in args:
        print_with_color(f"{arg_name} = {locals_dict[arg_name]}")

    # 处理可变位置参数 *args
    if varargs:
        varargs_values = locals_dict[varargs]
        for idx, value in enumerate(varargs_values):
            print_with_color(f"*{varargs}[{idx}] = {value}")

    # 处理可变关键字参数 **kwargs
    if varkw:
        kwargs_dict = locals_dict[varkw]
        for key, value in kwargs_dict.items():
            print_with_color(f"**{varkw} {key} = {value}")

'''
min-max归一化，将数值缩放到0-1区间
'''
def min_max_normalize(data):
    """Min-Max归一化"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除零