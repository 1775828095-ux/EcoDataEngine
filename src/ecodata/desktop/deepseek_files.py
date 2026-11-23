"""Approved native TIFF uploads to the official DeepSeek Files endpoint."""
import json
import ssl
from urllib.request import Request, build_opener
from uuid import uuid4

MAX_DOCUMENT_BYTES = 100 * 1024**2


def upload_tiff(stream, size, *, api_key, open_request=None):
    if not api_key or not 0 < size <= MAX_DOCUMENT_BYTES:
        raise ValueError("TIFF 大小或 API 授权不符合 Files API 要求")
    if open_request is None:
        import certifi
        from .deepseek_transport import APIHTTPSHandler
        from .service import NoRedirect

        open_request = build_opener(NoRedirect(), APIHTTPSHandler(
            context=ssl.create_default_context(cafile=certifi.where()))).open
    boundary = "ecodata-" + uuid4().hex

    def part(name, value):
        return f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'

    before = (part("purpose", "rich_layout_recognition") + part("expires_after[anchor]", "created_at")
              + part("expires_after[seconds]", "3600")
              + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="approved-raster.tif"\r\n'
              + "Content-Type: image/tiff\r\n\r\n").encode("ascii")
    after = f"\r\n--{boundary}--\r\n".encode("ascii")

    def body():
        yield before
        left = size
        while left:
            chunk = stream.read(min(left, 1024**2))
            if not chunk:
                raise ValueError("已批准的 TIFF 文件未读完整")
            left -= len(chunk)
            yield chunk
        yield after

    request = Request("https://api.deepseek.com/files", data=body(), method="POST", headers={
        "Authorization": "Bearer " + api_key,
        "Content-Type": "multipart/form-data; boundary=" + boundary,
        "Content-Length": str(len(before) + size + len(after)),
    })
    with open_request(request, timeout=245) as response:
        receipt = json.loads(response.read(1024 * 1024))
    identifier = receipt.get("id")
    if not isinstance(identifier, str) or not identifier.startswith("file-api-") or receipt.get("bytes") != size:
        raise ValueError("DeepSeek 未确认原生 TIFF 的上传回执")
    return dict(id=identifier, bytes=size, expires_at=receipt.get("expires_at"))
