import time, threading, markdown

'''
将markdown格式的内容转换为适合Qt中富文本控件展示的html格式内容
'''


def markdown_to_html(md_content):
    # 基础转换
    html = markdown.markdown(md_content)

    # 添加自定义CSS样式
    styled_html = f"""
    <style>
        .title {{ color: #2c3e50; font-weight: 600; margin-bottom: 8px; }}
        .list {{ margin-left: 20px; }}
        code {{ 
            background: #f8f9fa;
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }}
        pre {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 6px;
            white-space: pre-wrap;
        }}
    </style>
    {html}
    """
    return styled_html
