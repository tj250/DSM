from colorama import Fore, Style
from enum import Enum
from dataclasses import dataclass
import inspect, datetime

'''
土壤属性制图任务执行阶段
'''


class TaskStage(Enum):
    Uncertain = -1  # 尚未明确
    AcceptTask = 1  # 等待确定任务类型
    DataExplore = 2  # 数据探索
    BuildModel = 3  # 建模
    Evaluation = 4  # 评估
    Finished = 5  # 结束


class PredictionType(Enum):
    Raster = 1  # 栅格预测
    Vector = 2  # 矢量预测
    NonSpatial = 3  # 非空间表格数据预测


'''
会话消息类型
'''


class ConversationMessageType(Enum):
    Uncertain = -1  # 不确定类型
    SystemMessage = 1  # 系统消息
    AgentMessage = 2  # 由系统后台Agent所发出的消息（包括基于规则的以及LLM的消息包装后的）
    UserMessage = 3  # 用户在前端操作所形成的操作消息


'''
任务分析时，用户在界面上做出的选择项枚举
'''


class TaskAnalysisConfirmUIChoice(Enum):
    AnalyzingAgain = 0  # 重新对任务进行分析
    EnterNextStep = 1  # 进入下一个环节


'''
用于建模的数据探索(DataExploring)时，用户在界面上做出的选择项枚举
'''


class DEUIChoice(Enum):
    NoneChoice = "未选择"
    ChangeDataSource = "更换数据源"
    CorrectPredictionVariable = "直接指定响应变量"
    CorrectInterpretationVariable = "直接指定解释变量"
    CorrectCategoticalVariables = "直接指定类别型变量"
    EnterEsda = "确认分析结果"


'''
用于建模的数据探索(DataExploring)时，即将显示给用户的选择窗口的类型
'''


class DEUIType(Enum):
    Prediction = 0  # 选择进入"重新指定数据源，还是直接指定预测变量"的UI界面
    Categorical = 1  # 选择进入"重新指定数据源，还是直接指定类别变量"的UI界面


'''
建模进展状态
'''


class BMActiveState(Enum):
    Beginning = 1  # 尚未开始
    HeuristicsFiltered = 2  # 启发式过滤
    ParamsSearching = 3  # 最优参数搜索中
    ParamsSearched = 4  # 最优参数搜索结束
    ResultConfirmed = 5  # 用户确认参数搜索结果
    Stacking = 6  # 堆叠异构模型
    Finished = 7  # 已经结束
    Test = 8


'''
建模时，用户在界面上做出的选择项枚举
'''


class BMUIChoice(Enum):
    NoneChoice = -1
    Stacking = 0  # 堆叠模型以优化建模结果
    EnterNextStep = 1  # 进入下一个环节


'''
制图时，用户在界面上做出的选择项枚举
'''


class UserConfirmUIChoice(Enum):
    MappingAgain = "重新制图"  # 重新进行制图
    EvaluatingAgain = "重新评估"  # 重新进行评估
    Confirm = "确认结果"  # 确认制图或评估结果


'''
用于建模的数据探索(DataExploring)时，即将显示给用户的选择窗口的类型
'''


class UncertaintyType(Enum):
    STD = "标准差"                    # 标准差
    Percentile = "5%分位数"              # 分位数
    Confidence95 = "95%置信区间"            # 95%置信区间分布
    VariationCoefficient = "变异系数"    # 变异系数

class RegressionPredictionType(Enum):
    Raster = 1  # 栅格预测
    Vector = 2  # 矢量图像预测
    NonSpatial = 3  # 非空间的结构化数据预测


'''
带有色彩文本的枚举定义，不同Agent具有不同的色彩
'''


class AgentColor(Enum):
    UNSET = Fore.CYAN  # 未指定
    ACCEPTTASK = Fore.LIGHTBLUE_EX  # 接收任务
    DATAEXPLORER = Fore.YELLOW  # 数据探索
    BUILDMODEL = Fore.LIGHTGREEN_EX  # 建模
    EVALUATION = Fore.LIGHTYELLOW_EX  # 预测

    HUMAN = Fore.LIGHTWHITE_EX  # 人类反馈


'''
控制台输出带有色彩的文本
'''


def print_agent_output(output: str, agent: str = "UNSET"):
    # 生成时间戳（格式：[HH:MM:SS]）
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
    print(f"{timestamp} {AgentColor[agent].value}{agent}: {output}{Style.RESET_ALL}")


'''
控制台打印输出函数调用参数信息
'''


def print_caller_parameters():
    # 获取调用者的栈帧
    caller_frame = inspect.currentframe().f_back
    # 获取调用者的参数信息
    args, varargs, varkw, locals_dict = inspect.getargvalues(caller_frame)

    print_agent_output(f"函数 {caller_frame.f_code.co_name} 被调用，参数如下：", agent="EVALUATION")

    # 打印固定参数
    for arg_name in args:
        print_agent_output(f"{arg_name} = {locals_dict[arg_name]}")

    # 处理可变位置参数 *args
    if varargs:
        varargs_values = locals_dict[varargs]
        for idx, value in enumerate(varargs_values):
            print_agent_output(f"*{varargs}[{idx}] = {value}")

    # 处理可变关键字参数 **kwargs
    if varkw:
        kwargs_dict = locals_dict[varkw]
        for key, value in kwargs_dict.items():
            print_agent_output(f"**{varkw} {key} = {value}")


@dataclass
class InternalException:
    """系统执行过程中内部产生的异常"""

    agent: str  # 发生异常的agent名称
    description: str  # 异常描述
