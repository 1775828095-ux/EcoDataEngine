"""Logical, validation-only stopping; no test/confirmation feedback or BO replacement."""
from pathlib import Path
import json,hashlib,math

POLICY=Path(__file__).with_name('search_stop_policy.json')


def policy(request):
    declared=request.get('search_stopping')
    if declared is None:
        if POLICY.exists():
            content=POLICY.read_bytes();result=json.loads(content)
            result['sha256']=hashlib.sha256(content).hexdigest()
            result['calibration_scope_match']=request.get('variable')=='ndvi' and request.get('unit')=='1'
            return result
        return dict(enabled=False,evidence='No calibrated policy published')
    return dict(declared)


class SearchStopper:
    def __init__(self,declaration,resolution,*,unit='numerical_evaluation'):
        self.unit=unit;self.numerical_evaluations=0
        self.policy=declaration;self.enabled=declaration.get('enabled',True)
        self.resolution=resolution;self.reason=None;self.anchor=None;self.total=0;self.stale=0;self.empty=0;self.last=0
        if not self.enabled:return
        for name in ['min_evaluations','patience_evaluations','no_novel_iterations']:
            if type(declaration.get(name)) is not int or declaration[name]<1:raise ValueError('Positive integer stopping policy required: '+name)
        gain=declaration.get('min_relative_gain')
        if not isinstance(gain,(int,float)) or not math.isfinite(gain) or not 0<=gain<1:raise ValueError('Finite stopping relative gain in [0,1) required')
        target=declaration.get('target_error')
        if target is not None and (not isinstance(target,(int,float)) or not math.isfinite(target) or target<0):raise ValueError('Nonnegative finite internal-validation target required')

    def start(self,best):
        if best is not None:self.anchor=float(best['metrics']['objective'])
        if self.enabled and self.anchor is not None and self.policy.get('target_error') is not None and self.anchor<=self.policy['target_error']:
            self.reason='VALIDATION_TARGET_REACHED'

    def observe(self,evaluations,best,*,numerical_evaluations=None,empty_coverage_complete=True):
        self.numerical_evaluations=evaluations if numerical_evaluations is None else numerical_evaluations
        new=evaluations-self.last;self.last=evaluations;self.total+=new;self.stale+=new
        self.empty=0 if new else self.empty+1
        if best is not None:
            value=float(best['metrics']['objective'])
            if self.anchor is None or self.anchor-value>max(self.resolution,abs(self.anchor)*self.policy.get('min_relative_gain',0.)):
                self.anchor=value;self.stale=0
            if self.enabled and self.policy.get('target_error') is not None and value<=self.policy['target_error']:
                self.reason='VALIDATION_TARGET_REACHED'
        if not self.enabled or self.reason:return
        if self.empty>=self.policy['no_novel_iterations'] and empty_coverage_complete:self.reason='NO_NEW_NUMERICAL_EVALUATIONS'
        elif self.total>=self.policy['min_evaluations'] and self.stale>=self.policy['patience_evaluations']:
            self.reason='NO_MATERIAL_VALIDATION_GAIN'

    def receipt(self):
        return dict(policy=self.policy,reason=self.reason,new_numerical_evaluations=self.numerical_evaluations,
            stagnation_unit=self.unit,completed_stopping_units=self.total,
            evaluations_since_material_gain=self.stale,no_novel_iterations=self.empty,
            reference='Internal search validation only; test and independent confirmation unused',global_optimum_certified=False)
