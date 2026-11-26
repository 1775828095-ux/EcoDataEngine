"""Explicit saved scientific declarations, scoped to exact original bytes.

These records contain no algorithms. Working assumptions retain their label;
matching an archive fingerprint does not prove a historical scientific meaning.
"""
from pathlib import Path
import hashlib,io,json

STORE=Path(__file__).resolve().parents[3]/'data/approved_input_contexts'


def find(path,model=None):
    path=Path(path)
    if not path.is_file() or not STORE.is_dir():return None
    if model:
        # A context for a different model cannot apply; avoid hashing GB inputs
        # merely to discover that fact. Same-model matches still require SHA256.
        records=[]
        for candidate in STORE.glob('*.json'):
            try:records.append(json.loads(candidate.read_text(encoding='utf8')))
            except (OSError,ValueError):continue
        if not any(item.get('model_id')==model for item in records):return None
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    record=STORE/(sha+'.json')
    if not record.is_file():return None
    context=json.loads(record.read_text(encoding='utf8'))
    if context.get('schema')!='ecodata.saved-input-context.v1' or context.get('source_sha256')!=sha:
        raise ValueError('保存的输入上下文指纹/版本不符')
    if model and context.get('model_id')!=model:return None
    if context.get('status') not in {'WORKING_ASSUMPTION_ONLY','CONFIRMED_DECLARATION'}:
        raise ValueError('保存的输入上下文缺少明确证据状态')
    return context


def augment(uploads,model,progress=None):
    if len(uploads)!=1 or getattr(uploads[0],'path',None) is None:return uploads,None
    context=find(uploads[0].path,model)
    if not context:return uploads,None
    extra=[]
    for name,document in context.get('documents',{}).items():
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError('上下文文档路径不合法')
        if document.get('schema')!='ecodata.engineering-semantics.v1' and document.get('status')!='WORKING_ASSUMPTION_ONLY':
            raise ValueError('上下文仅接受已有科学语义声明及显式工作假设，不能注入处理方案')
        if document.get('status')=='WORKING_ASSUMPTION_ONLY' and document.get('schema'):
            raise ValueError('工作假设文档不能同时声明其他执行协议')
        stream=io.BytesIO(json.dumps(document,ensure_ascii=False).encode('utf8'));stream.name=name;extra.append(stream)
    if progress:progress('沿用与原始文件SHA256一致的历史科学声明；证据状态：'+context['status']+'；'+context.get('explanation',''))
    return [*uploads,*extra],{k:v for k,v in context.items() if k!='documents'}
