"""Independent MODIS red/NIR observations for hidden-date reconstruction."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from acquire_basin_ecology import catalog, crop, save, OUT


def main():
    jobs=[]
    for year in range(2013,2019):
        items=[i for i in catalog('modis-09Q1-061',year)
               if i['properties']['platform']=='terra']
        latest={}
        for i in items:
            p=i['properties'];key=(p['start_datetime'],p['modis:horizontal-tile'],p['modis:vertical-tile'])
            if key not in latest or i['id']>latest[key]['id']:latest[key]=i
        for i in latest.values():
            for key in ['sur_refl_b01','sur_refl_b02','sur_refl_qc_250m','sur_refl_state_250m']:
                jobs.append((i,key,'modis_reflectance'))
        print('CATALOG',year,len(latest),flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending={pool.submit(crop,*j):j for j in jobs}
        for n,f in enumerate(as_completed(pending),1):
            try:f.result()
            except Exception as exc:failures.append(dict(item=pending[f][0]['id'],asset=pending[f][1],error=str(exc)))
            if n%40==0:print('MODIS WINDOWS',n,'/',len(jobs),'failures',len(failures),flush=True)
    save(OUT/'halfdate_acquisition.json',dict(collection='modis-09Q1-061',planned=len(jobs),failures=failures,
         independent_sensor=True,months='full_year',through_year=2018,source='https://doi.org/10.5067/MODIS/MOD09Q1.061'))
    if failures:raise RuntimeError('Incomplete source intake; failures retained')


def fine():
    jobs=[]
    for year in range(2013,2019):
        items=[i for i in catalog('landsat-c2-l2',year)
               if i['properties'].get('platform')=='landsat-8'
               and i['properties'].get('landsat:collection_category')=='T1']
        for i in items:
            for key in ['red','nir08','qa_pixel','qa_radsat']:jobs.append((i,key,'landsat'))
        print('FINE CATALOG',year,len(items),flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending={pool.submit(crop,*j):j for j in jobs}
        for n,f in enumerate(as_completed(pending),1):
            try:f.result()
            except Exception as exc:failures.append(dict(item=pending[f][0]['id'],asset=pending[f][1],error=str(exc)))
            if n%80==0:print('FINE WINDOWS',n,'/',len(jobs),'failures',len(failures),flush=True)
    save(OUT/'halfdate_fine_acquisition.json',dict(planned=len(jobs),failures=failures,months='full_year',through_year=2018,selection='All Landsat8 T1 acquisitions, no effect selection'))
    if failures:raise RuntimeError('Fine intake incomplete')


if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='full':main();fine()
    elif len(sys.argv)>1 and sys.argv[1]=='fine':fine()
    else:main()
