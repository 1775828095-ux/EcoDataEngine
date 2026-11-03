"""Official LGHAP ZIP member range retrieval, CRC validation, no entire archive claim."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile
import requests


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


class RemoteZip(io.RawIOBase):
    def __init__(self,url,size,log):self.url=url;self.size=size;self.pos=0;self.log=log;self.session=requests.Session()
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def read(self,size=-1):
        size=self.size-self.pos if size<0 else min(size,self.size-self.pos)
        if size<=0:return b""
        start=self.pos;end=start+size-1
        for attempt in range(2):
            try:
                r=self.session.get(self.url,headers={"Range":f"bytes={start}-{end}"},timeout=(15,60),stream=True)
                with r:
                    if r.status_code!=206 or r.headers.get("Content-Range","").split("/")[0]!=f"bytes {start}-{end}":
                        raise ValueError(f"Range not honored: {r.status_code} {r.headers.get('Content-Range')}")
                    value=r.content
                if len(value)!=size:raise ValueError("Truncated HTTP range")
                self.log.append(dict(start=start,end=end,bytes=len(value),attempt=attempt+1,status=206))
                self.pos+=len(value);return value
            except Exception as exc:
                self.log.append(dict(start=start,end=end,attempt=attempt+1,error=str(exc)))
                if attempt:raise


def main(out,dates):
    out.mkdir(parents=True,exist_ok=True)
    api="https://zenodo.org/api/records/5652257"
    record=requests.get(api,timeout=30);record.raise_for_status();record=record.json()
    save(out/"official_record.json",record)
    files={f["key"]:f for f in record["files"]}
    guide=files["Data user guide and demo codes.zip"]
    path=out/guide["key"]
    if not path.exists():
        r=requests.get(guide["links"]["self"],timeout=60);r.raise_for_status();path.write_bytes(r.content)
    assert len(path.read_bytes())==guide["size"] and "md5:"+hashlib.md5(path.read_bytes()).hexdigest()==guide["checksum"]
    with zipfile.ZipFile(path) as z:
        save(out/"guide_members.json",[dict(name=i.filename,size=i.file_size) for i in z.infolist()])
        for i in z.infolist():
            if not i.is_dir() and Path(i.filename).suffix.lower() in {".txt",".py",".pdf"}:
                (out/Path(i.filename).name).write_bytes(z.read(i))
    yearly=files["LGHAP.AOD.D001.Y2019.zip"];log=[]
    try:
        remote=RemoteZip(yearly["links"]["self"],yearly["size"],log)
        with zipfile.ZipFile(remote) as z:
            entries=[dict(name=i.filename,size=i.file_size,compressed_size=i.compress_size,CRC32=f"{i.CRC:08x}") for i in z.infolist()]
            save(out/"year2019_members.json",entries)
            # First get one actual day to inspect metadata; later dates fixed before formal evaluation.
            members=[i for i in z.infolist() if not i.is_dir() and Path(i.filename).suffix.lower()==".nc"]
            requested=[i for i in members if any(date in i.filename for date in dates)] if dates else members[:1]
            if dates and len(requested)!=len(set(dates)):raise ValueError("Requested dates not unique in official ZIP")
            manifests=[]
            for i in requested:
                path=out/Path(i.filename).name
                if path.exists():
                    import zlib
                    content=path.read_bytes()
                    assert len(content)==i.file_size and zlib.crc32(content)==i.CRC
                else:
                    content=z.read(i)  # zipfile checks the official central-directory member CRC
                    path.write_bytes(content)
                manifests.append(dict(source=yearly["links"]["self"],archive_size=yearly["size"],
                    archive_official_md5=yearly["checksum"],whole_archive_downloaded=False,whole_archive_MD5_verified=False,
                    member=i.filename,member_bytes=len(content),member_CRC32=f"{i.CRC:08x}",member_CRC_verified=True,
                    sha256=hashlib.sha256(content).hexdigest(),verified_UTC=datetime.now(timezone.utc).isoformat()))
                print("VERIFIED",i.filename,len(content),flush=True)
            save(out/("requested_manifest.json" if dates else "sample_manifest.json"),manifests)
    finally:save(out/"range_requests.json",log)
    print([p.name for p in out.iterdir()],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dates",nargs="*",default=[])
    args=parser.parse_args();main(args.output,args.dates)
