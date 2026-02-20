from PyQt5.QtWidgets import QWidget, QVBoxLayout
from qgis.core import QgsRasterLayer, QgsProject
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtCore import Qt


class MapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        # 创建地图画布
        self.canvas = QgsMapCanvas()
        self.canvas.setCanvasColor(Qt.white)

        # 设置窗口布局
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def load_raster(self, file_path: str) -> bool:
        """加载栅格图层并自动适配显示范围"""
        layer = QgsRasterLayer(file_path, "Raster Layer", "gdal")
        if not layer.isValid():
            return False

        QgsProject.instance().addMapLayer(layer)
        self.canvas.setExtent(layer.extent())
        self.canvas.setLayers([layer])
        self.canvas.refreshAllLayers()
        return True


    def clear_map(self):
        """清除当前地图内容"""
        QgsProject.instance().removeAllMapLayers()
        self.canvas.refreshAllLayers()
