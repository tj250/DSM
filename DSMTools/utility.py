from colorama import Fore, Style
import inspect,datetime

def print_with_color(output:str, color:str=Fore.CYAN) -> None:
    # 生成时间戳（格式：[HH:MM:SS]）
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {color}{output}{Style.RESET_ALL}")

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