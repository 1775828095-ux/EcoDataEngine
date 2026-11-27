"""Bounded real old-desktop-entry acceptance; no independent accuracy claim."""
import argparse,hashlib,json
from pathlib import Path
import xarray as xr
from ecodata.algorithms.platform_search import SearchInput
from ecodata.desktop.file_agent import scan_index
from ecodata.desktop.selected_processing import run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    source=Path(args.input);folder=Path(args.output)
    if folder.exists():raise ValueError('New immutable output directory required')
    folder.mkdir(parents=True)
    content=source.read_bytes();fingerprint=hashlib.sha256(content).hexdigest()
    with xr.open_dataset(source) as data:
        variable=next(iter(data.data_vars));unit=data[variable].attrs['units']
        physical=data[variable].attrs['physical_variable']
    request=dict(task='multisource',variable=variable,physical_variable=physical,unit=unit,
        semantic_type='continuous',temporal_semantics='instantaneous',processing_goal='spatial_fusion',
        observation_crs='EPSG:4326',mutation_control='random')
    state=dict(mode='fusion',fmt='NetCDF',variable=variable,source_formats=['NetCDF'],
        items=[SearchInput(source.name,'primary',content)],request=request,eligible=['spatial.idw'])
    task=dict(action='fusion',state=state,recipes=[['spatial.idw']],autonomous_recommendation=True,
        autonomous_variables=[variable],route='local')
    scan,_=scan_index(source)
    result=run(scan,[('C1:fusion',task)],folder/'run',print,model_id='data_only')
    products=[]
    for name,raw in result.get('files',[]):
        (folder/name).write_bytes(raw)
        if name.endswith('.nc'):
            with xr.open_dataset(folder/name) as restored:
                products.append(dict(name=name,dimensions=dict(restored.sizes),finite=bool(restored.to_array().notnull().all())))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==fingerprint
    failures=result.get('failed_tasks',[])
    receipt=dict(input=str(source),sha256=fingerprint,unchanged=True,products=products,failures=failures,
        scope='Actual legacy desktop handler with online prepared DS recommendation; not manual GUI acceptance')
    (folder/'acceptance.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
    if not products or failures:raise RuntimeError('Old entry acceptance failed; retained receipt')


if __name__=='__main__':main()
