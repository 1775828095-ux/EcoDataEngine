"""Hide prior job results when the material inputs or explicit choices change."""
import hashlib


def bind_result(state,job_key,binding_key,*values):
    digest=hashlib.sha256()
    for value in values:
        raw=value if isinstance(value,bytes) else str(value).encode('utf8')
        digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    fingerprint=digest.hexdigest();old=state.get(binding_key)
    if old!=fingerprint:state.pop(job_key,None)
    state[binding_key]=fingerprint
    return fingerprint
