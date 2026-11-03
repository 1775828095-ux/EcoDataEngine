"""Independent numerical/protocol audit, without calling the scoring engine."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,math,sys
import traceback
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/(sys.argv[1] if len(sys.argv)>1 else "output/complementary_20261024/study_v2")


def load(p):return json.loads(p.read_text(encoding="utf-8"))


def observation_matrix(fine,coarse):
    def edges(x):
        x=np.asarray(x,float)
        return np.r_[x[0]-(x[1]-x[0])/2,(x[:-1]+x[1:])/2,x[-1]+(x[-1]-x[-2])/2]
    def weights(f,c,latitude):
        f,c=edges(f),edges(c)
        if latitude:f,c=np.sin(np.deg2rad(f)),np.sin(np.deg2rad(c))
        overlap=np.maximum(0,np.minimum(c[1:,None],f[None,1:])-np.maximum(c[:-1,None],f[None,:-1]))
        return overlap/np.diff(c)[:,None]
    return np.kron(weights(fine.lat,coarse.lat,True),weights(fine.lon,coarse.lon,False))


def audit_training(profile):
    """Rebuild all training errors and second moments from saved actual products."""
    parent=ROOT/'output/complementary_20261024/study_v2'
    count=0
    for group_index,start in enumerate(profile['training_groups']):
        folder=parent/'oracles'/str(start);report=load(folder/'summary.json')
        request=load(folder/'request.json');var=request['request']['variable']
        fine_path=next(i['path'] for i in request['visible_input_snapshots'] if i['source_id']=='fine_slow')
        with xr.open_dataset(fine_path,engine='h5netcdf') as ds:scale=float(ds[var].std())
        with xr.open_dataset(parent/'cases'/str(start)/'hidden_reference.nc',engine='h5netcdf') as ds:truth=ds[var].load()
        recorded={r['key']:r for r in load(folder/'independent_matrix.json')['rows']};buckets=defaultdict(list)
        for row in report['rows']:
            assert row['status']=='PASS'
            path=Path(row['product']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['product_sha256']
            with xr.open_dataset(path,engine='h5netcdf') as ds:e=(ds[var]-truth).values.astype(float)
            np.testing.assert_allclose(np.sqrt(np.mean(e*e)),recorded[row['key']]['rmse'],atol=1e-9)
            plan=row['plan'];fusion=next(p['choice'] for p in plan if p['branch']=='fusion')
            alpha=fusion['detail_strength'] if fusion['algorithm']=='fusion.spatiotemporal_detail' else fusion['primary_share'] if fusion['primary']=='fine_slow' else 1-fusion['primary_share']
            family=dict(branches=[dict(branch=p['branch'],choice=p['choice']) for p in plan if p['branch'] not in {'fusion','post'}],fusion=fusion['algorithm'])
            key=hashlib.sha256(json.dumps(family,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
            buckets[key].append((alpha,e/scale));count+=1
        for key,items in buckets.items():
            (a,e),(b,f)=sorted(items,key=lambda i:i[0])[:2]
            slope=(f-e)/(b-a);eh=e+(1-a)*slope;ec=e-a*slope
            measured=np.array([[np.mean(eh*eh),np.mean(eh*ec)],[np.mean(eh*ec),np.mean(ec*ec)]])
            np.testing.assert_allclose(measured,profile['families'][key]['second_moments'][group_index],atol=1e-8)
        print(json.dumps(dict(training_group=start,independently_checked_products=count)),flush=True)
    return count


def main():
    result=load(OUT/"results.json")
    followup=(OUT/'frozen_protocol.json').exists()
    if followup:
        selection=load(OUT/'frozen_protocol.json');assert not selection['fresh_test_seen_at_freeze']
        assert selection['development']==[0,360,720,1080]
        assert selection['fresh_test']==[1600,1960,2320,2680]
        training_products=audit_training(load(OUT/'response_portrait.json'))
    else:
        selection=load(OUT/"frozen_selection.json");assert selection["test_seen"] is False
    assert all(s<1460 for s in [0,360,720,1080,1200,1340])
    failures=[];products=0;screened=0;runs=0;coarse_max=0.;case_data={};groups=defaultdict(list)
    for folder in sorted(p.parent for p in OUT.rglob("summary.json")):
        try:
            report=load(folder/"summary.json");protocol=load(folder/"request.json")
            screen=load(folder/"initial_screening.json")
            assert screen["complete"] and screen["sampling"] is False
            assert screen["scored_count"]+screen["rejected_count"]==protocol["grammar_count"]
            assert len(screen["rows"])==screen["scored_count"]
            order=sorted(screen["rows"],key=lambda r:(-round(r["score_interval"][0],9),r["scientific_steps"],r["index"]))
            assert [r["index"] for r in order[:2]]==screen["survivors"]
            # Identify date group from raw visible snapshots, not directory assumptions.
            snaps={r["source_id"]:r for r in protocol["visible_input_snapshots"]}
            sources={}
            for key,item in snaps.items():
                p=Path(item["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
                with xr.open_dataset(p,engine="h5netcdf") as ds:sources[key]=ds.load()
            fine=sources["fine_slow"];coarse=sources["coarse_fast"];var=protocol["request"]["variable"]
            target=pd.DatetimeIndex(protocol["request"]["target_times"])
            assert fine.sizes["time"]==2 and coarse.sizes["time"]==9
            assert not target.isin(pd.DatetimeIndex(fine.time)).any()
            stamp=str(fine.time.values[0]);o=observation_matrix(fine,coarse)
            h=fine[var].interp(time=target).transpose("time","lat","lon").values
            c=coarse[var].sel(time=target).transpose("time","lat","lon").values.reshape(len(target),-1)
            gram=o@o.T
            def lift(a):return np.linalg.solve(gram,a.T).T@o
            hflat=h.reshape(len(target),-1);hd=hflat-lift(hflat@o.T)
            baseline=dict(coarse_error=float(np.sqrt(np.mean((hflat@o.T-c)**2))),detail_error=float(np.sqrt(np.mean(hd**2))))
            response=protocol.get('response_portrait_snapshot')
            declaration=protocol['request']['composition'].get('response_profile')
            if declaration:
                raw=Path(declaration['path']).read_bytes()
                assert hashlib.sha256(raw).hexdigest()==declaration['sha256']
                saved_response=json.loads(raw)
                if response is not None:assert response==saved_response
                response=saved_response
            if response:
                baseline['expected_error']=1.
                assert response['training_partition']=='development' and response['training_groups']==[0,360,720,1080]
                for source in response['training_sources']:
                    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
            matching_model=protocol["portraits"][0]["model"]
            issues={i["attribute"]:i for i in matching_model["state_policy"]["scenario"]["issues"]}
            for row in report["rows"]:
                if row["status"]!="PASS":
                    assert row.get("error");continue
                p=Path(row["product"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==row["product_sha256"]
                with xr.open_dataset(p,engine="h5netcdf") as ds:output=ds.load()
                assert np.isfinite(output[var]).all() and output[var].attrs["units"]=="K"
                y=output[var].transpose("time","lat","lon").values.reshape(len(target),-1)
                metrics=dict(coarse_error=float(np.sqrt(np.mean((y@o.T-c)**2))),detail_error=float(np.sqrt(np.mean(((y-lift(y@o.T))-hd)**2))))
                if response:
                    plan=row['plan'];fusion=next(p['choice'] for p in plan if p['branch']=='fusion')
                    a=fusion['detail_strength'] if fusion['algorithm']=='fusion.spatiotemporal_detail' else fusion['primary_share'] if fusion['primary']=='fine_slow' else 1-fusion['primary_share']
                    family=dict(branches=[dict(branch=p['branch'],choice=p['choice']) for p in plan if p['branch'] not in {'fusion','post'}],fusion=fusion['algorithm'])
                    key=hashlib.sha256(json.dumps(family,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
                    moments=np.asarray(response['families'][key]['second_moments']).mean(axis=0);v=np.array([a,1-a])
                    metrics['expected_error']=float(np.sqrt(v@moments@v))
                value=0.
                for term in row["matching"]["fields"][0]["terms"]:
                    issue=issues[term["dimension"]];metric=issue["metric"]
                    np.testing.assert_allclose(term["data_evidence"]["before"],baseline[metric],atol=1e-9)
                    np.testing.assert_allclose(term["algorithm_evidence"]["after"],metrics[metric],atol=1e-8)
                    def risk(q):return float(np.clip((q-issue["acceptable"])/(issue["bad"]-issue["acceptable"]),0,1))
                    contribution=issue['importance']/sum(i['importance'] for i in issues.values())*(risk(baseline[metric])-risk(metrics[metric]))
                    np.testing.assert_allclose(term["interval"],[contribution,contribution],atol=1e-8);value+=contribution
                np.testing.assert_allclose(row["matching"]["score_interval"],[value,value],atol=1e-8)
                if any(p["branch"]=="fusion" and p["choice"]["algorithm"]=="fusion.spatiotemporal_detail" for p in row["plan"]):
                    coarse_max=max(coarse_max,metrics["coarse_error"]);assert metrics["coarse_error"]<1e-8
                products+=1
            valid=[r for r in report["rows"] if r["status"]=="PASS"]
            expected=min(valid,key=lambda r:(-round(r["matching"]["score_interval"][0],9),r["scientific_steps"],r["index"]))
            assert expected["key"]==report["recommended_key"]
            assert not report["reference_used_for_selection"]
            partition=report.get("evaluation_protocol",{}).get("study_partition")
            if partition=="test":assert pd.Timestamp(stamp).year==2014
            elif partition in {"development","validation"}:assert pd.Timestamp(stamp).year==2013
            # Recompute independent selected error using a separate saved target file.
            group=next(p for p in (OUT/"cases").iterdir() if pd.DatetimeIndex(load(p/"lineage.json")["hidden_times"]).equals(target))
            with xr.open_dataset(group/"hidden_reference.nc",engine="h5netcdf") as ds:truth=ds[var].load()
            with xr.open_dataset(expected["product"],engine="h5netcdf") as ds:
                e=ds[var]-truth;rmse=float(np.sqrt(np.mean(e.values**2)))
            assert np.isclose(report["independent_evaluation"]["RMSE"],rmse)
            groups[(stamp,partition)].append(tuple(screen["survivors"]))
            screened+=screen["scored_count"];runs+=1
        except Exception as exc:failures.append(dict(folder=str(folder),error=str(exc),traceback=traceback.format_exc()))
    # Identical task/pool requests across backends/seeds have identical initial seeds.
    # Source-only controls intentionally have a different candidate domain.
    assert runs>=(90 if followup else 100) and not failures,failures[:5]
    audit=dict(status="PASS",runs=runs,products=products,full_scored_candidates=screened,
        projection_max_coarse_rmse=coarse_max,failures=failures,
        scope="Independent matrix construction, risk arithmetic, hashes, time masking, selection and independent error; no simulator claim")
    if followup:audit['independently_checked_training_products']=training_products
    save=OUT/'independent_audit.json'
    save.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(audit,ensure_ascii=False))


if __name__=="__main__":main()
