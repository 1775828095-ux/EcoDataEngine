"""Call unchanged starfm4py kernels with per-job isolated global parameters."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import importlib.util,json,sys,types
from pathlib import Path
import numpy as np
import dask.array as da
from numpy.lib.stride_tricks import sliding_window_view


def run():
    p=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));w=p['window_pixels']
    parameters=types.ModuleType('parameters')
    for k,v in dict(windowSize=w,logWeight=False,temp=False,mid_idx=w*w//2,numberClass=4,
        spatImp=p['spatial_impact_pixels'],specUncertainty=np.hypot(p['fine_uncertainty'],p['coarse_uncertainty']),
        tempUncertainty=np.sqrt(2)*p['coarse_uncertainty'],path='').items():setattr(parameters,k,v)
    sys.modules['parameters']=parameters
    path=Path(__file__).resolve().parents[1]/'external/starfm4py/src/starfm4py.py'
    spec=importlib.util.spec_from_file_location('official_starfm',path);upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    result=[]
    with np.load(sys.argv[1]) as data:
        for fine,anchor,target in zip(data['fine'],data['coarse_anchor'],data['coarse_target']):
            # Official source pads the image with zero and excludes zeros in
            # its similarity rule. Preserve this convention, with bounded tiles.
            padded=[np.pad(a,w//2,mode='constant') for a in [fine,anchor,target]]
            views=[sliding_window_view(a,(w,w)) for a in padded];rows=[]
            # Bound temporary window memory by rows, independently of graph
            # search budgets; no candidate or scientific step is omitted.
            block_rows=max(1,1048576//(fine.shape[1]*w*w))
            for y in range(0,fine.shape[0],block_rows):
                count=min(block_rows,fine.shape[0]-y)
                arrays=[da.from_array(a[y:y+count].reshape(count*fine.shape[1],w*w),chunks=(count*fine.shape[1],w*w)) for a in views]
                rows.append(upstream.predict(*arrays,(count,fine.shape[1])).compute(scheduler='single-threaded'))
            result.append(np.concatenate(rows,axis=0))
    np.save(sys.argv[3],np.stack(result))


if __name__=='__main__':run()
