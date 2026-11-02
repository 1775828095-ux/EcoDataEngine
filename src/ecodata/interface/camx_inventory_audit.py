"""Derive inventory coverage from actual native mapping, never a fixed count."""
import re


def audit_inventory(source_species, diagnostic, require_complete=False):
    if not source_species or len(set(source_species)) != len(source_species):
        raise ValueError('排放物种为空或存在重复')
    result={}
    for kind,tag in [('Area','AREA'),('Point','PT')]:
        pairs=re.findall(rf'{kind} source species\s+\d+\s+(\S+)\s+mapped to model species\s+\d+\s+(\S+)',diagnostic)
        mapped=dict(pairs)
        if not pairs or len(mapped)!=len(pairs) or not set(mapped)<=set(source_species):
            raise ValueError('原生排放映射缺失、重复或与源清单不一致：'+kind)
        ignored=set(re.findall(rf'{tag} species:\s*(\S+)\s+not modeled',diagnostic))
        missing=set(source_species)-set(mapped)
        if ignored!=missing:
            raise ValueError('清单物种未被映射且无明确原生日志解释：'+kind)
        result[kind]={'source_count':len(source_species),'mapped_count':len(mapped),
                      'mapping':mapped,'not_modeled':sorted(missing),'complete':not missing}
    result['complete']=all(result[k]['complete'] for k in ('Area','Point'))
    if require_complete and not result['complete']:
        raise ValueError('完整排放适配未通过；原生未建模物种：'+','.join(sorted(set(result['Area']['not_modeled']+result['Point']['not_modeled']))))
    return result
