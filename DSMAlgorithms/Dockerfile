FROM continuumio/miniconda3 AS builder
RUN conda install -c conda-forge gdal -y && conda clean -afy

FROM ac2-registry.cn-hangzhou.cr.aliyuncs.com/ac2/base:ubuntu24.04-py312-optimised
# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8389

# 启动命令（Flask 开发模式）
CMD ["python", "app.py"]
