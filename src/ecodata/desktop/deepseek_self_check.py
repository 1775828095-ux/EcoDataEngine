"""Verify the real TIFF and chat path with isolated synthetic bytes."""
import json
from io import BytesIO


def check(api_key, progress=None, *, upload=None, send=None):
    if not api_key or not api_key.strip():
        raise ValueError("联机自检需要已配置的 DeepSeek API 密钥")

    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from .deepseek_files import upload_tiff
    from .service import tool_turn

    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=128, height=128, count=1, dtype="uint8",
                         transform=from_origin(0, 1, 0.001, 0.001), crs="EPSG:4326") as raster:
            values = np.zeros((128, 128), dtype=np.uint8)
            values[20:108, 20:108] = 255
            raster.write(values, 1)
        memory.seek(0)
        synthetic = memory.read()
    if progress:
        progress("正在上传程序生成的合成 GeoTIFF")
    receipt = (upload or upload_tiff)(BytesIO(synthetic), len(synthetic), api_key=api_key)
    identifier = receipt["id"]
    messages = [dict(role="system", content="只返回 JSON 对象，键为 synthetic_sum，值为整数。"),
                dict(role="user", content=[
                    dict(type="file", file_id=identifier),
                    dict(type="text", text="这是接口连通性自检。请计算合成 CSV 中数值列之和：\nvalue\n1\n2"),
                ])]
    message, chat = (send or tool_turn)(messages, [], api_key, progress)
    try:
        observed = json.loads(message.get("content", ""))
    except (ValueError, TypeError) as error:
        raise ValueError("附件请求已响应，但返回的数据不符合自检格式") from error
    value = observed.get("synthetic_sum") if isinstance(observed, dict) else None
    if type(value) is not int or value != 3:
        raise ValueError("附件请求已响应，但合成 CSV 校验没有通过")
    return dict(status="passed", file_id=identifier, file_bytes=len(synthetic), chat_id=chat.get("id"),
                expires_at=receipt.get("expires_at"))
