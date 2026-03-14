#!/usr/bin/env python3
"""
使用 Claude 多模态模型分析图片
"""
import base64
from anthropic import Anthropic

API_KEY = "sk-sp-50b8bf7df4454e97b3e702f650ec9d85"
API_BASE = "https://coding.dashscope.aliyuncs.com/apps/anthropic"

IMAGE_PATH = "/Users/yimg/.nanobot/media/avatar-portfolio/nanobot-id-photo.png"

# 读取图片
with open(IMAGE_PATH, "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

client = Anthropic(api_key=API_KEY, base_url=API_BASE)

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data
                    }
                },
                {
                    "type": "text",
                    "text": "请详细描述这张图片：人物外貌特征、表情、发型、服装、背景、光线、整体风格。用中文回答。"
                }
            ]
        }
    ]
)

print(response.content[0].text)
