"""Separate graph evaluations, cache-only traversals and new numerical calls."""
import json
from collections import Counter
from run_robit_era5_mutation import OUT,load
from ecodata.algorithms.numerical_search import save


def main():
    rows=[]
    for folder in sorted((OUT/'runs').iterdir()):
        s=load(folder/'summary.json');last=0;counts=Counter();operators=Counter()
        for line in (folder/'attempts.jsonl').read_text(encoding='utf-8').splitlines():
            r=json.loads(line);delta=r['adapter_calls_total']-last;last=r['adapter_calls_total']
            assert delta>=0
            counts['graph_evaluations']+=1
            counts['new_adapter_calls']+=delta
            if delta:counts['graphs_with_new_numeric_calls']+=1
            elif r['status']=='EXECUTION_FAILED':counts['graphs_rejected_without_new_numeric_calls']+=1
            else:counts['graphs_fully_cached_or_raw_source']+=1
            if r['origin']=='mutation':
                counts['mutation_graph_evaluations']+=1
                category=('mutation_graphs_with_new_numeric_calls' if delta else
                          'mutation_rejected_without_new_numeric_calls' if r['status']=='EXECUTION_FAILED' else 'mutation_fully_cached_graphs')
                counts[category]+=1
            counts['failed_graphs']+=r['status']=='EXECUTION_FAILED'
        assert counts['graph_evaluations']==s['executed_candidates']
        assert counts['new_adapter_calls']==s['adapter_invocations']
        assert counts['mutation_graph_evaluations']==s['mutation_new_executions']
        for t in load(folder/'mutation_trace.json'):operators[t['edit']['operator']]+=1
        rows.append(dict(run=folder.name,**counts,operators=dict(operators),
                         deployment_adapter_calls=s['deployment_adapter_calls'],service_seconds=s['elapsed_seconds']))
    save(OUT/'cost_readback.json',dict(status='PASS',rows=rows,
      budget_basis='Upper limit on distinct graph evaluation requests; cache-only DAG traversals are not new numerical executions',
      actual_numeric_cost='New adapter invocations plus deployment, with expensive xsdba fits separately in execution_costs.json',
      xsdba_classification_limit='execution_costs.json classifies retained successful traces only; failed graphs can omit upstream traces, so xsdba counts are lower bounds, not complete fit totals',
      no_efficiency_advantage_assumed=True))
    print('COST READBACK PASS',len(rows))


if __name__=='__main__':main()
