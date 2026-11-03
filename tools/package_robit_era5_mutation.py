"""Archive read-back evidence without changing frozen scientific results."""
from pathlib import Path
import hashlib,json,zipfile
from run_robit_era5_mutation import ROOT,OUT,OLD,load,freeze


def main():
    freeze()
    for name in ['independent_audit.json','weather_readback.json','cost_readback.json','delivery_numerical_readback.json']:
        assert load(OUT/name)['status']=='PASS',name
    files={}
    def add(path):
        if path.is_file():files[path.relative_to(ROOT).as_posix()]=path
    for folder in ['runs','cases','predictions','source_frozen','weather_handoff','diagnostic_v6']:
        for path in (OUT/folder).rglob('*'):add(path)
    for path in OUT.iterdir():
        if path.suffix in {'.json','.xml','.csv','.md'} and path.name not in {'DELIVERY_MANIFEST.json','delivery_verification.json'}:add(path)
    for folder in sorted(OUT.glob('diagnostic*')):
        for path in folder.rglob('*'):
            if path.name in {'summary.json','mutation_trace.json','failure.json','initial_screening.json'}:add(path)
    for folder in ['cases','predictions','runs_v6','source_frozen','weather_handoff']:
        for path in (OLD/folder).rglob('*'):add(path)
    for path in OLD.iterdir():
        if path.suffix in {'.json','.nc','.xml'} and path.name not in {'DELIVERY_MANIFEST.json','delivery_verification.json'}:add(path)
    for directory in ['data/robit_era5_1990_2013','output/robit_single_source_20261026/source']:
        for path in (ROOT/directory).rglob('*'):add(path)
    for path in (ROOT/'output').glob('robit_era5_mutation*.log'):add(path)
    for path in (ROOT/'tools').glob('*robit*era5*.py'):add(path)
    add(ROOT/'tools/audit_robit_mutation_costs.py')
    add(ROOT/'tools/check_robit_mutation_delivery.py')
    for name in ['tests/test_field_mutation.py','pyproject.toml','requirements-composition-search.txt','requirements-lock.txt',
                 'docs/Robit_ERA5变异补跑_协议_20261028.md','docs/Robit_ERA5变异补跑_交付_20261028.md',
                 'docs/Robit_ERA5降水融合_实验交付_20261027.md','docs/EcoData Engine项目总交付报告.md']:
        add(ROOT/name)
    readme=OUT/'DELIVERY_README.md'
    readme.write_text('''# Robit＋ERA5变异机制补跑（2026-10-28）

先读docs/Robit_ERA5变异补跑_交付_20261028.md。正式结果只用runs；diagnostic与diagnostic_v2—v5是开发诊断，旧轨迹保留但不合并计效果。diagnostic_v6含最终自动加载模型策略和可见值自遮挡的公共入口验收。详细早期诊断日志/完整运行仍在原项目目录，未删除。

顺序为三画像筛选→既有ALNS库的开放图适配→真实DAG执行与验证误差反馈→冻结预测→离线测试。未换成GOLEM或开放枚举。画像未知保持UNKNOWN，不能将它称为已学得SWAT响应。实际搜索空间未闭合，无全局最优认证。

已看过上一轮测试，因此这是同数据补跑，非新的盲测。新冻结预测与上一轮固定、单源、随机全部保留。标签是离线评价资源，禁止传入算法输入。input_hashes.json记录沿用原输入指纹；source_manifest.json和source_frozen为冻结科学源码，当前文件哈希核对通过。

预算按图评估请求计；完全缓存图不算新增数值计算，详见cost_readback.json。新增适配器调用、部署和xsdba拟合成本分列；不能拿候选数直接声称计算优势。

weather_handoff仅9套新天气试验版本（3遮挡×ALNS/同邻域随机/仅画像初始化seed11），每套2192日。上一轮15套及全部预测也保留。未运行SWAT，不含完整流域工程，不应覆盖队友file.cio/WGN或完整原观测。

这是现有EcoDataEngine项目的证据/源码包，不含虚拟环境。复现需项目依赖及composition-search（alns 7.0.0）；环境见environment_versions.json。路径保持原绝对路径，异机须在副本中重定位，勿在冻结交付目录重新prepare或predict覆盖证据。先查看运行器和协议，再决定是否全量重算。

ZIP内所有证据文件的字节数和SHA256见DELIVERY_MANIFEST.json，交付前已逐成员回读校验。压缩包自身校验结果位于原输出目录delivery_verification.json。
''',encoding='utf-8')
    add(readme)
    manifest={name:dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name,path in sorted(files.items())}
    manifest_path=OUT/'DELIVERY_MANIFEST.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8');add(manifest_path)
    archive_path=OUT/'Robit_ERA5变异补跑_交付_20261028.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name,row in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==row['sha256'],name
        assert 'ALNS' in archive.read('docs/Robit_ERA5变异补跑_交付_20261028.md').decode('utf-8')
    receipt=dict(status='PASS',files=len(files),bytes=archive_path.stat().st_size,sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                 all_member_hashes_checked=True,zip=str(archive_path))
    (OUT/'delivery_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
