"""Package existing frozen evidence; never rerun selection or score here."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/robit_era5_fusion_20261027'


def main():
    files = {}

    def add(path):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path

    for name in ['independent_audit.json', 'weather_readback.json']:
        assert json.loads((OUT / name).read_text(encoding='utf-8'))['status'] == 'PASS'
    for folder in ['runs_v6', 'predictions', 'source_frozen', 'cases', 'weather_handoff']:
        for path in (OUT / folder).rglob('*'):
            add(path)
    for path in OUT.iterdir():
        if path.suffix in {'.json', '.nc', '.xml'} and path.name not in {'DELIVERY_MANIFEST.json', 'delivery_verification.json'}:
            add(path)
    for path in (ROOT / 'data/robit_era5_1990_2013').rglob('*'):
        add(path)
    for path in (ROOT / 'output/robit_single_source_20261026/source').rglob('*'):
        add(path)
    for name in ['Robit_ERA5降水融合_实验交付_20261027.md', 'Robit_ERA5融合_冻结实验协议_20261027.md',
                 'EcoData Engine项目总交付报告.md']:
        add(ROOT / 'docs' / name)
    for name in ['acquire_robit_era5.py', 'run_robit_era5_fusion.py', 'evaluate_robit_era5_fusion.py',
                 'audit_robit_era5_fusion.py', 'check_robit_era5_weather.py', 'package_robit_era5_delivery.py',
                 'run_robit_single_source.py', 'evaluate_open_complementary.py']:
        add(ROOT / 'tools' / name)
    add(ROOT / 'tests/test_series_fusion.py')
    for pattern in ['pyproject.toml', 'requirements*.txt', '*lock*']:
        for path in ROOT.glob(pattern):
            add(path)
    for path in (ROOT / 'output').glob('robit_era5*.log'):
        add(path)
    readme = OUT / 'DELIVERY_README.md'
    readme.write_text('''# Robit ERA5交付包（2026-10-28）

先读docs/Robit_ERA5降水融合_实验交付_20261027.md。正式证据为runs_v6；旧协议和失败日志用于记录诊断，不计正式效果。详细V1—V5运行与缓存仍保存在原项目输出目录，未为缩小交付包而删除。

自动相对站点单源平均RMSE改善3.36%，但对最强固定略差0.04%，无可靠搜索优势。原始天气完整，本轮为受控遮挡恢复，不是校正原实测真值；不含SWAT完整工程、径流或全局最优认证。

所有文件由DELIVERY_MANIFEST.json列明SHA256。source_frozen为正式运行时科学源码；tools中保留当前导出日期索引及审计路径修复，详见export_only_patch.json。其他源码按冻结版本，不要用诊断版替换。

此包是可审计证据与源码快照，非带依赖安装器的独立应用。复现需现有EcoDataEngine项目与environment_versions.json相应环境。配置保留原绝对路径；在副本中运行须统一重定位输入/输出路径，勿在本交付目录执行prepare覆盖已冻结证据。重算评分/审计会加载离线标签，在线推荐输入不得加入reference_DO_NOT_PASS_TO_ALGORITHMS.nc或test_labels.nc。

weather_handoff含15套2008—2013天气试验包，每套2192日。只包含天气与索引，不覆盖file.cio、WGN或流域拓扑。自动/随机天气样本为预定seed11，非挑选最好种子；全部种子NetCDF另在predictions。不要覆盖完整原观测。尚未运行SWAT。
''', encoding='utf-8')
    add(readme)
    manifest = {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
                for name, path in sorted(files.items())}
    manifest_path = OUT / 'DELIVERY_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    add(manifest_path)
    destination = OUT / 'Robit_ERA5融合_实验交付_20261027.zip'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        for name, row in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == row['sha256'], name
        assert '未证明自动组合' in archive.read('docs/Robit_ERA5降水融合_实验交付_20261027.md').decode('utf-8')
    result = dict(status='PASS', files=len(files), bytes=destination.stat().st_size,
                  zip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                  all_member_hashes_rechecked=True, zip=str(destination))
    (OUT / 'delivery_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
