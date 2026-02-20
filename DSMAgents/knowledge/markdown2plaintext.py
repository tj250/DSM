import markdown

# 读取 Markdown 文件
with open('input.md', 'r', encoding='utf-8') as f:
    markdown_text = f.read()

# 转换为纯文本
plain_text = markdown.markdown(markdown_text, extensions=['markdown.extensions.extra'])

# 保存为纯文本文件
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(plain_text)
