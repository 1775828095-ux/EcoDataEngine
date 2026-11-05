"""Local-path intake must not allocate the full input archive."""
import hashlib
from ecodata.interface.engineering_upload import LocalUpload
from ecodata.interface.workbench_page import input_identity


def test_local_identity_matches_upload_without_eager_read(tmp_path):
    path = tmp_path / '原始资料.zip'
    raw = b'unchanged source archive bytes'
    path.write_bytes(raw)

    class StreamingOnlyLocal(LocalUpload):
        def getvalue(self):
            raise AssertionError('Local files must be streamed, not loaded whole')

    class Uploaded:
        name = path.name
        def getvalue(self):
            return raw

    original = hashlib.sha256(path.read_bytes()).hexdigest()
    assert input_identity([StreamingOnlyLocal(path.name, path)], 'CAMx') == input_identity([Uploaded()], 'CAMx')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original
    path.write_bytes(raw + b'changed')
    assert input_identity([StreamingOnlyLocal(path.name, path)], 'CAMx') != input_identity([Uploaded()], 'CAMx')
