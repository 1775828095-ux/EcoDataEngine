"""Final acceptance gate over tests, frozen code and independently saved evidence."""
from importlib.metadata import version
from pathlib import Path
import hashlib,json,shutil,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/parallel_portrait_20261025'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit=read(OUT/'independent_audit.json');entry=read(OUT/'delivery/acceptance.json')
    assert audit['status']==entry['status']=='PASS'
    tests=set();receipts=[]
    for name in ('parallel_scoring_compatibility.xml','parallel_scoring_regression.xml'):
        path=ROOT/'output'/name;tree=ET.parse(path)
        for tc in tree.iter('testcase'):
            assert tc.find('failure') is None and tc.find('error') is None
            tests.add((tc.attrib.get('classname'),tc.attrib['name']))
        receipts.append(dict(path=str(path),sha256=sha(path)))
    assert len(tests)==71
    manifest=read(OUT/'source_manifest.json')
    for item in manifest:
        path=Path(item['path']);assert sha(path)==item['sha256']
        assert sha(OUT/'source_frozen'/path.relative_to(ROOT))==item['sha256']
    extra=[ROOT/'tools'/n for n in ('audit_parallel_portrait_study.py','accept_parallel_portrait_delivery.py',
        'summarize_parallel_portrait_study.py','finalize_parallel_portrait_delivery.py')]
    extra.append(ROOT/'tests/test_parallel_composition_scoring.py')
    released=[]
    for path in [Path(x['path']) for x in manifest]+extra:
        target=OUT/'source_release'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        released.append(dict(path=str(path),sha256=sha(path)))
    evidence=[]
    for name in ('protocol.json','calibration.json','frozen_selections.json','results.json','independent_audit.json','delivery/acceptance.json','case_results.csv'):
        path=OUT/name;evidence.append(dict(path=str(path),sha256=sha(path)))
    report=ROOT/'docs/三路并行评分_多区域实验交付_20261025.md';assert report.is_file()
    result=dict(status='PASS',unique_tests=len(tests),test_receipts=receipts,
        frozen_scientific_sources_unchanged=True,released_sources=released,evidence=evidence,
        report=dict(path=str(report),sha256=sha(report)),
        products_independently_checked=audit['real_numeric_products_checked'],
        test_products_independently_checked=audit['test_products_checked'],
        package_versions={p:version(p) for p in ('numpy','scipy','xarray','h5py','h5netcdf','alns','pytest','streamlit')},
        scope='Acceptance of implementation and frozen experiment, not a guarantee of favourable results or universal downstream response')
    (OUT/'final_assurance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',unique_tests=len(tests),products=audit['real_numeric_products_checked'],test_products=audit['test_products_checked'])),flush=True)


if __name__=='__main__':main()
