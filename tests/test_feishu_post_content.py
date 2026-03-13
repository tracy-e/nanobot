from nanobot.channels.feishu import _extract_post_text


def test_extract_post_text_localized_format() -> None:
    """Feishu post content in localized format: {"zh_cn": {"title": ..., "content": ...}}"""
    payload = {
        "zh_cn": {
            "title": "日报",
            "content": [
                [
                    {"tag": "text", "text": "完成"},
                    {"tag": "img", "image_key": "img_1"},
                ]
            ],
        }
    }

    text = _extract_post_text(payload)

    assert text == "日报 完成"


def test_extract_post_text_direct_format() -> None:
    """Feishu post content in direct format: {"title": ..., "content": ...}"""
    payload = {
        "title": "Daily",
        "content": [
            [
                {"tag": "text", "text": "report"},
                {"tag": "img", "image_key": "img_a"},
                {"tag": "img", "image_key": "img_b"},
            ]
        ],
    }

    text = _extract_post_text(payload)

    assert text == "Daily report"


