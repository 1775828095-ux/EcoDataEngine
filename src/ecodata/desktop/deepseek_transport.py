"""Streaming official API transport with progress and bounded transient retries."""
from contextlib import contextmanager
from http.client import HTTPSConnection, IncompleteRead, RemoteDisconnected
import json
import socket
import ssl
import tempfile
from pathlib import Path
from threading import Event, Thread
import time
import urllib.error
import urllib.request

CONNECT_TIMEOUT = 20
READ_TIMEOUT = 180
TOTAL_TIMEOUT = 900
ATTEMPTS = 3
STAGES = {"file_classification":"文件分类", "decision":"资料判断", "recommendation":"算法推荐",
          "fusion_review":"融合复核", "output_check":"输出检查"}


class ConnectTimeout(TimeoutError):
    pass


class APIConnection(HTTPSConnection):
    def __init__(self,*args,read_timeout=None,**kwargs):
        self.read_timeout=READ_TIMEOUT if read_timeout is None else read_timeout
        super().__init__(*args,**kwargs)

    def connect(self):
        try:
            super().connect()
        except TimeoutError as error:
            raise ConnectTimeout() from error
        self.sock.settimeout(self.read_timeout)


class APIHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        timeout=getattr(request,'ecodata_read_timeout',None)
        factory=APIConnection if timeout is None else lambda *a,**k:APIConnection(*a,read_timeout=timeout,**k)
        return self.do_open(factory, request, context=self._context)


class StreamInterrupted(OSError):
    pass


@contextmanager
def waiting(progress, stage, attempt, *, attempt_limit=None):
    stopped = Event()
    started = time.monotonic()
    phase = ["正在连接官网"]
    prefix = f"DeepSeek {STAGES.get(stage,stage)} · 第 {attempt}/{ATTEMPTS if attempt_limit is None else attempt_limit} 次尝试"

    def emit():
        progress(f"{prefix} · {phase[0]} · 已等待 {int(time.monotonic()-started)} 秒")

    def update(value):
        if value != phase[0]:
            phase[0] = value
            emit()

    def heartbeat():
        while not stopped.wait(5):
            emit()

    emit()
    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield update
    finally:
        stopped.set()
        worker.join()


def read_reply(response, update, *, total_timeout=None):
    started = time.monotonic()
    content = []
    tool_calls = {}
    result = {"choices":[]}
    finish = None
    first = True
    while True:
        if time.monotonic()-started > (TOTAL_TIMEOUT if total_timeout is None else total_timeout):
            raise TimeoutError("total_response_timeout")
        raw = response.readline()
        if not raw:
            raise StreamInterrupted("stream_closed_before_done")
        line = raw.strip()
        if not line:
            continue
        # Preserve compatibility with gateways returning a complete JSON response.
        if first and line.startswith(b"{"):
            return json.loads(raw+response.read())
        first = False
        if line.startswith(b":"):
            continue
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            if finish is None:
                raise StreamInterrupted("stream_missing_finish_status")
            message = {"role":"assistant", "content":"".join(content) or None}
            if tool_calls:
                message['tool_calls'] = [tool_calls[i] for i in sorted(tool_calls)]
            result["choices"] = [{"finish_reason":finish,"message":message}]
            return result
        chunk = json.loads(data)
        if chunk.get("error"):
            raise ValueError("DeepSeek 流式返回错误，未收到完整结果")
        for key in ("id","model","usage"):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        for choice in chunk.get("choices",[]):
            if choice.get("index",0) != 0:
                continue
            delta = choice.get("delta",{})
            for part in delta.get('tool_calls', []):
                call = tool_calls.setdefault(part['index'], dict(id='', type='function', function=dict(name='', arguments='')))
                if part.get('id'):
                    call['id'] = part['id']
                function = part.get('function', {})
                call['function']['name'] += function.get('name', '')
                call['function']['arguments'] += function.get('arguments', '')
            if delta.get("content"):
                update("正在生成结果")
                content.append(delta["content"])
            elif delta.get("reasoning_content") and not content:
                update("已连接，正在思考")
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]


def failure_info(error, connected):
    reason = error.reason if isinstance(error,urllib.error.URLError) else error
    if isinstance(reason,ssl.SSLCertVerificationError):
        return "TLS 证书验证失败",False
    if isinstance(reason,ssl.SSLError):
        return "TLS 安全连接失败",False
    if isinstance(reason,socket.gaierror):
        return "DNS 域名解析失败",True
    if isinstance(reason,ConnectTimeout):
        return f"连接或代理握手超时（等待 {CONNECT_TIMEOUT} 秒）",True
    if isinstance(reason,TimeoutError):
        if str(reason)=="total_response_timeout":
            return f"回复总等待超过 {TOTAL_TIMEOUT} 秒",True
        return (f"等待官网回复超时（连续 {READ_TIMEOUT} 秒未收到数据）" if connected
                else f"发送请求或等待官网响应头超时（等待 {READ_TIMEOUT} 秒）"),True
    if isinstance(reason,ConnectionRefusedError):
        return "连接被拒绝，请检查系统代理端口或网络",True
    if isinstance(reason,(StreamInterrupted,IncompleteRead,RemoteDisconnected,ConnectionResetError,BrokenPipeError)):
        return "连接中途断开，未收到完整回复",True
    if str(reason).startswith("Tunnel connection failed:"):
        return "系统代理无法建立到 DeepSeek 的安全隧道",True
    code = getattr(reason,"winerror",None) or getattr(reason,"errno",None)
    return f"网络连接失败（{type(reason).__name__}"+(f"，错误码 {code}" if code is not None else "")+"）",True


def record_failure(stage, digest, attempt, elapsed, error, message):
    path = Path(tempfile.gettempdir())/"ecodata-deepseek-network.jsonl"
    record = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),stage=stage,request_digest=digest,
                  attempt=attempt,elapsed_seconds=round(elapsed,2),error_type=type(error).__name__,
                  http_status=getattr(error,"code",None),message=message)
    reason = error.reason if isinstance(error,urllib.error.URLError) else error
    record.update(reason_type=type(reason).__name__,errno=getattr(reason,"errno",None),
                  winerror=getattr(reason,"winerror",None),verify_code=getattr(reason,"verify_code",None))
    try:
        with path.open("a",encoding="utf8") as file:
            file.write(json.dumps(record,ensure_ascii=False)+"\n")
    except OSError:
        pass


def fetch(request, opener, stage, digest, progress, *, attempts=None,total_timeout=None):
    overall = time.monotonic()
    limit=ATTEMPTS if attempts is None else attempts
    if total_timeout is not None:request.ecodata_read_timeout=min(READ_TIMEOUT,total_timeout)
    for attempt in range(1,limit+1):
        connected = False
        started = time.monotonic()
        try:
            with waiting(progress,stage,attempt,attempt_limit=limit) as update:
                with opener.open(request,timeout=CONNECT_TIMEOUT) as response:
                    connected = True
                    update("已连接，等待思考结果")
                    if total_timeout is not None:
                        sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
                        if sock is not None:sock.settimeout(min(READ_TIMEOUT,total_timeout))
                        return read_reply(response,update,total_timeout=total_timeout)
                    return read_reply(response,update)
        except urllib.error.HTTPError as error:
            retry = error.code in {429,500,502,503,504}
            message = f"官网返回 HTTP {error.code}"
            record_failure(stage,digest,attempt,time.monotonic()-started,error,message)
            if not retry or attempt==limit:
                error.attempt_count = attempt
                error.elapsed_seconds = round(time.monotonic()-overall,1)
                raise
            error.close()
        except (urllib.error.URLError,OSError,IncompleteRead) as error:
            message,retry = failure_info(error,connected)
            record_failure(stage,digest,attempt,time.monotonic()-started,error,message)
            if not retry or attempt==limit:
                raise ValueError(f"DeepSeek {message}；共尝试 {attempt} 次，耗时 {time.monotonic()-overall:.1f} 秒。") from None
        delay = attempt*2
        progress(f"DeepSeek {message} · {delay} 秒后自动重试（第 {attempt+1}/{ATTEMPTS} 次）")
        time.sleep(delay)
