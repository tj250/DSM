from graphviz import Digraph
from langgraph.graph import START, StateGraph, END
import re

def save_graph_to_png(graph: StateGraph, filename: str):
    try:
        # 尝试获取图结构（可能需要根据实际graph对象调整属性）
        visualization = visualize_state_graph(graph)

        # 在Jupyter中直接显示
        output_path = visualization.render(
            filename=filename,  # 文件名（无需后缀）
            format='png',  # 指定格式
            directory='./agents/graph_structure',  # 可选：保存目录（默认当前路径）
            cleanup=True  # 自动清理临时文件
        )
    except AttributeError as e:
        print(f"需要根据实际graph对象结构调整属性访问方式，当前错误: {e}")
    except Exception as e:
        print(f"请确保已安装graphviz: {e}")


'''
对图进行可视化
'''


def visualize_state_graph(graph) -> Digraph:
    dot = Digraph(comment='State Graph', encoding='UTF-8')
    dot.attr('graph', fontname='SimSun')  # 设置图形字体为宋体:ml-citation{ref="2,4" data="citationList"}
    dot.attr('node', fontname='SimSun')  # 节点标签字体
    dot.attr('edge', fontname='SimSun')  # 边标签字体

    # 提取节点和边信息（需根据实际graph结构调整）
    nodes = graph.nodes  # 如果graph有nodes属性
    edges = graph.edges  # 如果graph有edges属性

    # 添加节点（这里假设nodes是字典或列表）
    for node in nodes:
        if node is START or node is END:
            dot.node(str(node),
                     "开始" if node is START else "结束",
                     color="black",  # 可选颜色区分
                     fontcolor="black",  # 标签颜色
                     fontsize="12"
                     )
        else:
            dot.node(str(node),
                     split_text_by_length(str(node)),
                     color="darkgreen",  # 可选颜色区分
                     fontcolor="darkgreen",  # 标签颜色
                     fontsize="12",
                     shape='rect',
                     style='rounded'
                     )

    # 遍历边并识别条件
    for edge in edges:
        if edge.conditional:
            # 条件分支边（渲染为虚线）
            dot.edge(
                edge.source, edge.target,
                label=edge.data,
                style="dashed",         # 关键样式设置
                color="blue",           # 可选颜色区分
                fontcolor="blue",  # 标签颜色
                fontsize="10"
            )
        else:
            # 普通边（实线）
            dot.edge(edge.source,
                     edge.target,
                     color="darkgreen"
            )
    return dot

def split_text_by_length(text, chunk_size=6):
    """
    将长文本按指定长度分隔并用换行符连接
    :param text: 输入文本
    :param chunk_size: 分隔长度，默认为6
    :return: 分隔后的文本
    """
    return '\n'.join([text[i:i+chunk_size] for i in range(0, len(text), chunk_size)])