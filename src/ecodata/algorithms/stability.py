"""Post-selection, truth-free source-offset stress test using SALib Morris.

This measures response, not error, uncertainty calibration or source usefulness.
No perturbation bounds are inferred to be measurement uncertainties.
"""
from pathlib import Path
from time import perf_counter
from importlib.metadata import version
import hashlib,io
import numpy as np
import xarray as xr
from .numerical_search import save


def assess(inputs,request,result,folder,*,execute):
    from .platform_search import parse_inputs,SearchInput
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    report=dict(protocol='source_offset_morris_v1',status='NOT_EVALUATED',
        used_for_selection=False,accuracy_assessed=False,uncertainty_calibrated=False,
        perturbation='uniform additive source offset: up to 1% visible source standard deviation; restricted by declared nonnegative input domain',
        perturbation_basis='fixed engineering stress, NOT a measured instrument error bound',
        scope='source-common-mode offsets only; not arbitrary pixel noise, geographic errors or proof of accuracy',
        seed=11,trajectories=2,levels=4,max_sources=3,max_executions=8,
        wall_policy='60 seconds checked between executions; running kernels are not forcibly interrupted',
        sources=[],excluded=[],runs=[],new_executions=0,failures=0,cache_hits=0)
    try:
        variable=request['variable'];graph=result.get('selected_graph')
        if not graph or not result.get('product'):
            report['reason']='No frozen executable graph/product';return report
        import json
        report['graph_sha256']=hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest()
        report['baseline_product_sha256']=hashlib.sha256(Path(result['product']).read_bytes()).hexdigest()
        if request.get('semantic_type')!='continuous':
            report['reason']='Additive numeric stress is not valid for categorical semantics';return report
        try:
            from SALib.sample.morris import sample
            from SALib.analyze.morris import analyze
        except ImportError:
            report['reason']='Optional SALib dependency unavailable';return report
        report['backend']='SALib '+version('SALib')
        # References, labels, observations, training states and target coordinates
        # never become perturbation factors. Keep their exact original bytes.
        active=[i for i,item in enumerate(inputs) if item.role in {'primary','background','auxiliary'}]
        report['excluded']=[dict(index=i,role=item.role,reason='Protected reference/state/observation/target role; unchanged')
            for i,item in enumerate(inputs) if i not in active]
        parsed,_=parse_inputs([inputs[i] for i in active]);datasets={}
        for i,(_,ds) in zip(active,parsed):
            if variable not in ds:
                report['excluded'].append(dict(index=i,reason='Different-variable auxiliary is not covered by this target-variable stress protocol'));continue
            a=ds[variable]
            if a.dtype.kind not in 'fiu' or not a.attrs.get('units'):
                report['excluded'].append(dict(index=i,reason='Non-numeric or missing units'));continue
            finite=np.isfinite(a.values);scale=float(np.std(a.values[finite])) if finite.any() else 0.
            if not np.isfinite(scale) or scale<=0:
                report['excluded'].append(dict(index=i,reason='No nonzero finite visible variation; no scale invented'));continue
            if any(s['input_sha256']==hashlib.sha256(inputs[i].content).hexdigest() for s in report['sources']):
                report['reason']='Duplicate source content cannot be treated as independent perturbation factors';return report
            datasets[i]=ds
            report['sources'].append(dict(index=i,role=inputs[i].role,unit=a.attrs['units'],
                source_id=ds.attrs.get('source_id'),input_sha256=hashlib.sha256(inputs[i].content).hexdigest(),
                offset_scale=.01*scale,finite_values=int(finite.sum()),
                factor_bounds=[max(-1.,-float(a.values[finite].min())/(.01*scale)) if request.get('nonnegative') else -1.,1.]))
        report['sources'].sort(key=lambda s:(s['role'],str(s['source_id']),s['input_sha256']))
        factors=report['sources'];d=len(factors)
        if not 1<=d<=3:
            report['reason']='Requires 1-3 varying, unit-labelled target-variable sources; none silently truncated';return report
        problem=dict(num_vars=d,names=[str(s['index']) for s in factors],bounds=[s['factor_bounds'] for s in factors])
        X=sample(problem,N=2,num_levels=4,seed=11)
        with xr.open_dataset(result['product']) as ds:base=ds[variable].load()
        report['response_unit']=base.attrs.get('units','UNKNOWN')
        mask=np.isfinite(base.values)
        if not mask.any():raise ValueError('No finite baseline output')
        responses=[];complete=True
        request=dict(request);request.pop('decision_policy',None)
        for j,offsets in enumerate(X):
            if perf_counter()-start>60:
                report['reason']='Post-selection wall budget exhausted';complete=False;break
            changed=list(inputs);row=dict(index=j,offsets=offsets.tolist(),status='FAILED')
            begin=perf_counter()
            try:
                for s,multiplier in zip(factors,offsets):
                    i=s['index'];ds=datasets[i].copy(deep=True);a=ds[variable]
                    a.data=np.where(np.isfinite(a.values),a.values+multiplier*s['offset_scale'],a.values)
                    buffer=io.BytesIO();ds.to_netcdf(buffer,engine='h5netcdf')
                    changed[i]=SearchInput('source.nc',inputs[i].role,buffer.getvalue(),inputs[i].metadata)
                report['new_executions']+=1
                deployed=execute(changed,request,graph,folder/str(j))
                with xr.open_dataset(deployed['product']) as ds:out=ds[variable].load()
                xr.testing.assert_equal(base.coords,out.coords)
                if out.dims!=base.dims or out.shape!=base.shape or out.attrs.get('units')!=base.attrs.get('units'):
                    raise ValueError('Perturbed output support/unit changed')
                if not np.isfinite(out.values[mask]).all():raise ValueError('Baseline finite coverage lost')
                difference=out.values[mask]-base.values[mask]
                response=float(np.mean(out.values[mask]));responses.append(response)
                row.update(status='EXECUTED',mean_output=response,response_RMS=float(np.sqrt(np.mean(difference**2))),
                    max_absolute_change=float(np.max(abs(difference))),negative_outputs=int((out.values[mask]<0).sum()),
                    product=deployed['product'],execution_cost=deployed.get('cost'))
            except Exception as exc:
                complete=False;report['failures']+=1;row['error']=str(exc)
            row['seconds']=perf_counter()-begin;report['runs'].append(row)
        if complete and len(responses)==len(X):
            indices=analyze(problem,X,np.asarray(responses),num_levels=4,seed=11)
            report['mean_output_sensitivity']=[dict(index=s['index'],mu_star=float(mu),sigma=float(sig))
                for s,mu,sig in zip(factors,indices['mu_star'],indices['sigma'])]
            report['status']='DIAGNOSTIC_COMPLETE'
        else:report['status']='INCOMPLETE'
        return report
    except Exception as exc:
        report.update(status='NOT_EVALUATED',reason=str(exc));return report
    finally:
        report['seconds']=perf_counter()-start
        save(folder/'stability.json',report)


def attach(inputs,request,result,folder,execute):
    """Common post-selection consumer; never changes graph, fitness or product."""
    folder=Path(folder)
    report=assess(inputs,request,result,folder/'stability',execute=execute)
    result['post_selection_stability']=report
    save(folder/'summary.json',result)
    return result
