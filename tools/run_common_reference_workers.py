"""Resource-only dispatcher; jobs and frozen scientific code are unchanged."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import sys
from run_common_reference_study import specs,METHODS,SEEDS,search_job,oracle_job,freeze


if __name__=='__main__':
    is_oracle=len(sys.argv)>1 and sys.argv[1]=='oracles'
    jobs=[x['id'] for x in specs()] if is_oracle else [(x['id'],m,s) for x in specs() for m in METHODS for s in SEEDS]
    function=oracle_job if is_oracle else search_job
    with ProcessPoolExecutor(max_workers=4 if is_oracle else 12) as pool:
        for f in as_completed([pool.submit(function,x) for x in jobs]):
            print(json.dumps(dict(stage=function.__name__,completed=f.result())),flush=True)
    if not is_oracle:freeze()
