"""Store provider credentials in Windows Credential Manager, never project files."""
import ctypes
from ctypes import wintypes
import os

TARGET = 'EcoDataEngine/DeepSeek'


class _Credential(ctypes.Structure):
    _fields_ = [('Flags',wintypes.DWORD),('Type',wintypes.DWORD),('TargetName',wintypes.LPWSTR),
                ('Comment',wintypes.LPWSTR),('LastWritten',wintypes.FILETIME),
                ('CredentialBlobSize',wintypes.DWORD),('CredentialBlob',ctypes.POINTER(ctypes.c_ubyte)),
                ('Persist',wintypes.DWORD),('AttributeCount',wintypes.DWORD),
                ('Attributes',ctypes.c_void_p),('TargetAlias',wintypes.LPWSTR),('UserName',wintypes.LPWSTR)]


def _api():
    if os.name != 'nt':
        raise OSError('当前系统未接入安全密钥存储；仅在本次会话使用')
    api=ctypes.WinDLL('advapi32',use_last_error=True)
    api.CredWriteW.argtypes=[ctypes.POINTER(_Credential),wintypes.DWORD]
    api.CredWriteW.restype=wintypes.BOOL
    api.CredReadW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                           ctypes.POINTER(ctypes.POINTER(_Credential))]
    api.CredReadW.restype=wintypes.BOOL
    api.CredFree.argtypes=[ctypes.c_void_p]
    api.CredDeleteW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD]
    api.CredDeleteW.restype=wintypes.BOOL
    return api


def load(target=TARGET):
    api=_api();pointer=ctypes.POINTER(_Credential)()
    if not api.CredReadW(target,1,0,ctypes.byref(pointer)):
        code=ctypes.get_last_error()
        if code==1168:return ''
        raise OSError(f'系统密钥读取失败（代码{code}）')
    try:
        value=pointer.contents
        return ctypes.string_at(value.CredentialBlob,value.CredentialBlobSize).decode('utf8')
    finally:api.CredFree(pointer)


def save(value,target=TARGET):
    api=_api();raw=value.encode('utf8')
    if not raw or len(raw)>2560:raise ValueError('密钥为空或超过系统存储长度')
    blob=(ctypes.c_ubyte*len(raw)).from_buffer_copy(raw)
    credential=_Credential(Type=1,TargetName=target,CredentialBlobSize=len(raw),CredentialBlob=blob,
                           Persist=2,UserName='DeepSeek')
    if not api.CredWriteW(ctypes.byref(credential),0):
        raise OSError(f'系统密钥保存失败（代码{ctypes.get_last_error()}）')


def forget(target=TARGET):
    api=_api()
    if not api.CredDeleteW(target,1,0) and ctypes.get_last_error()!=1168:
        raise OSError('系统密钥删除失败')
