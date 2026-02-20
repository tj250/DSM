import sys
sys.path.append(r"D:\PycharmProjects\DSMAlgorithms")
from PyQt5.QtWidgets import QWidget, QListWidget, QVBoxLayout,QHBoxLayout, QLabel,QToolBar, QAction,QComboBox
from PyQt5.QtGui import QIcon
from qgis.core import *
from qgis.gui import QgsMapCanvas,QgsMapToolPan, QgsMapToolZoom, QgsMapToolIdentify
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from data_access.mapping import get_raster_map_file,get_uncertainty_raster_map_file
from DSMAlgorithms.base.base_data_structure import algorithms_dict, AlgorithmsType
from agents.utils.views import UncertaintyType

'''
主窗口中反馈区域的widget
'''


class FeedbackWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.parent = parent
        self.layout = QVBoxLayout(self) # 创建一个默认的垂直布局
        self.setLayout(self.layout)
        self.canvas = None
        self.algorithms_id = None
        self.map_types = None
        self.setFixedWidth(800)

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()  # 安全删除控件
            elif item.layout():
                self.clear_layout(item.layout())  # 递归清除子布局
            del item  # 删除布局项
    def clear_content(self):
        if self.canvas is not None: # 如果之前显示了地图
            QgsProject.instance().removeAllMapLayers()
            self.canvas.refreshAllLayers()
            self.canvas.refresh()
            self.canvas = None
            self.clear_layout(self.layout) # 清除布局


    '''
    显示不确定性分析制图的结果
    '''
    def show_uncertainty_mapping_result(self, algorithms_id: str, algorithms_type: AlgorithmsType | list[AlgorithmsType], map_types:list[UncertaintyType]):
        self.algorithms_id = algorithms_id
        self.map_types = map_types
        # 清除旧内容
        self.clear_content()
        algorithms_name = "和".join([algorithms_dict[item] for item in algorithms_type]) if isinstance(algorithms_type, list) else algorithms_dict[algorithms_type]
        # 设置新的窗口布局
        # 1-加入显示标题
        title_label = QLabel(f"基于{algorithms_name}的不确定性制图结果预览，请在下方选择需要查看的不确定制图的类型", self)
        self.layout.addWidget(title_label)

        # 1-加入不确定性制图类型的列表
        self.combo = QComboBox()
        self.combo.addItems([map_type.value for map_type in map_types])  # 批量添加选项
        # 绑定选项变更信号
        self.combo.currentIndexChanged.connect(self.on_map_type_selection_change)
        self.layout.addWidget(self.combo)

        # 创建地图画布
        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)
        self.canvas.enableAntiAliasing(True)
        # 初始化地图工具
        self.init_map_tools()
        # 设置默认工具为漫游
        self.canvas.setMapTool(self.pan_tool)
        # 2-创建工具栏
        self.create_toolbar()
        # 3-地图窗口加入布局
        self.layout.addWidget(self.canvas)

        self.on_map_type_selection_change(0)

    def on_map_type_selection_change(self, index):
        map_type = self.map_types[index]

        QgsProject.instance().removeAllMapLayers()
        # 获取图层文件
        tif_file = get_uncertainty_raster_map_file(self.algorithms_id, map_type)

        """加载栅格图层并自动适配显示范围"""
        layer = QgsRasterLayer(tif_file, "Raster Layer", "gdal")
        if not layer.isValid():
            return

        stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        min_val, max_val = stats.minimumValue, stats.maximumValue

        # 配置着色器
        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        color_ramp_items = [
            QgsColorRampShader.ColorRampItem(min_val, QColor(61, 161, 209), "Min"),  # 蓝色
            QgsColorRampShader.ColorRampItem(min_val + (max_val-min_val)/2, QColor(241, 251, 123), "50%"),
            QgsColorRampShader.ColorRampItem(max_val, QColor(240, 38, 28), "Max")  # 红色
        ]
        color_ramp.setColorRampItemList(color_ramp_items)
        shader.setRasterShaderFunction(color_ramp)
        # 创建伪色彩渲染器
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader) # 1表示使用第一个波段
        layer.setRenderer(renderer)
        # layer.triggerRepaint()
        QgsProject.instance().addMapLayer(layer)
        self.canvas.setExtent(layer.extent())
        self.canvas.setLayers([layer])
        self.canvas.refreshAllLayers()

    '''
    显示某一制图的结果
    '''

    def show_mapping_result(self, algorithms_id: str, algorithms_type: AlgorithmsType | list[AlgorithmsType]):
        # 清除旧内容
        self.clear_content()
        algorithms_name = "和".join([algorithms_dict[item] for item in algorithms_type]) if isinstance(algorithms_type, list) else algorithms_dict[algorithms_type]
        # 设置新的窗口布局
        # 1-加入显示标题
        title_label = QLabel(f"基于{algorithms_name}的制图结果预览", self)
        self.layout.addWidget(title_label)

        # 创建地图画布
        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)
        self.canvas.enableAntiAliasing(True)
        # 初始化地图工具
        self.init_map_tools()
        # 设置默认工具为漫游
        self.canvas.setMapTool(self.pan_tool)
        # 2-创建工具栏
        self.create_toolbar()
        # 3-地图窗口加入布局
        self.layout.addWidget(self.canvas)

        # 获取图层文件
        tif_file = get_raster_map_file(algorithms_id)

        """加载栅格图层并自动适配显示范围"""
        layer = QgsRasterLayer(tif_file, "Raster Layer", "gdal")
        if not layer.isValid():
            return

        stats = layer.dataProvider().bandStatistics(1, QgsRasterBandStats.All)
        min_val, max_val = stats.minimumValue, stats.maximumValue

        # 配置着色器
        shader = QgsRasterShader()
        color_ramp = QgsColorRampShader()
        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
        color_ramp_items = [
            QgsColorRampShader.ColorRampItem(min_val, QColor(61, 161, 209), "Min"),  # 蓝色
            QgsColorRampShader.ColorRampItem(min_val + (max_val-min_val)/2, QColor(241, 251, 123), "50%"),
            QgsColorRampShader.ColorRampItem(max_val, QColor(240, 38, 28), "Max")  # 红色
        ]
        color_ramp.setColorRampItemList(color_ramp_items)
        shader.setRasterShaderFunction(color_ramp)
        # 创建伪色彩渲染器
        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader) # 1表示使用第一个波段
        layer.setRenderer(renderer)
        # layer.triggerRepaint()
        QgsProject.instance().addMapLayer(layer)
        self.canvas.setExtent(layer.extent())
        self.canvas.setLayers([layer])
        self.canvas.refreshAllLayers()

    def init_map_tools(self):
        # 创建漫游工具
        self.pan_tool = QgsMapToolPan(self.canvas)

        # 创建放大工具(左键放大)
        self.zoom_in_tool = QgsMapToolZoom(self.canvas, False)  # False表示放大

        # 创建缩小工具(左键缩小)
        self.zoom_out_tool = QgsMapToolZoom(self.canvas, True)  # True表示缩小

        # 创建Identify工具
        self.identify_tool = RasterValueTool(self.canvas, self.show_identify_result)

    def create_toolbar(self):
        # 创建工具栏对象
        self.toolbar = QToolBar()
        tool_layout = QHBoxLayout(self)  # 创建一个默认的垂直布局
        tool_layout.addWidget(self.toolbar, stretch=0)
        self.identify_label = QLabel("")
        tool_layout.addWidget(self.identify_label, stretch=1)
        self.layout.addLayout(tool_layout)

        # 添加工具按钮
        self.add_tool_button("./qt_ui/icons/zoom_in.png", "放大", self.on_zoom_in)
        self.add_tool_button("./qt_ui/icons/zoom_out.png", "缩小", self.on_zoom_out)
        self.add_tool_button("./qt_ui/icons/pan.png", "平移", self.on_pan)
        self.add_tool_button("./qt_ui/icons/full_extent.png", "重置地图显示范围", self.on_full_exent)
        self.add_tool_button("./qt_ui/icons/identify.png", "Identify", self.on_identify)

    def add_tool_button(self, icon_path, tooltip, callback):
        action = QAction(QIcon(icon_path), tooltip, self)
        action.triggered.connect(callback)
        action.setToolTip(tooltip)
        self.toolbar.addAction(action)

    def on_zoom_in(self):
        self.canvas.setMapTool(self.zoom_in_tool)

    def on_zoom_out(self):
        self.canvas.setMapTool(self.zoom_out_tool)

    def on_pan(self):
        self.canvas.setMapTool(self.pan_tool)
    def on_full_exent(self):
        extent = self.get_combined_extent()
        if extent.width() > 0 and extent.height() > 0:
            self.canvas.setExtent(extent)
            self.canvas.refresh()

    def on_identify(self):
        self.canvas.setMapTool(self.identify_tool)

    def show_identify_result(self, value:str):
        self.identify_label.setText(value)

    '''
    获取地图窗口图层边界
    '''
    def get_combined_extent(self):
        """
        获取当前项目中所有图层的联合范围
        :return: QgsRectangle对象表示的最大范围
        """
        # 初始化一个空的范围对象
        combined_extent = QgsRectangle()
        combined_extent.setMinimal()

        # 获取所有图层
        layers = QgsProject.instance().mapLayers().values()

        for layer in layers:
            if layer.isValid() and layer.extent().width() > 0 and layer.extent().height() > 0:
                # 合并每个图层的有效范围
                combined_extent.combineExtentWith(layer.extent())

        return combined_extent

class RasterValueTool(QgsMapToolIdentify):
    def __init__(self, canvas, identify_callback):
        super().__init__(canvas)
        self.setCursor(Qt.CrossCursor)
        self.identify_callback = identify_callback

    def canvasReleaseEvent(self, event):
        # 转换点击坐标至地图坐标
        # point = self.toMapCoordinates(event.pos())
        # 执行栅格值查询
        results = self.identify(event.pos().x(), event.pos().y(), QgsProject.instance().mapLayers().values())
        if results:
            # 输出第一个波段的栅格值
            print(f"坐标({event.pos().x():.4f}, {event.pos().y():.4f})处栅格值: {results[0].mAttributes['Band 1']}")
            self.identify_callback(results[0].mAttributes['Band 1'])