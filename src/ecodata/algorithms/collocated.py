"""Local residual update at evidenced collocations, not spatial extrapolation.

Reuse registered residual kernels and the same typed executor/quality gates.
There is one admissible route under this contract, no accuracy optimiser.
"""
import numpy as np
from .dag import Artifact,Port,Node,NodeAdapter,NumericalDAGExecutor
from .registry import AlgorithmRegistry
from ecodata.fusion.residual import residual_difference,residual_addition
from ecodata.evaluation.quality_gates import RawInputQualityGate,ResultQualityGate


def local_residual(background,observed,sampled,variable):
    field=background[variable];observed=np.asarray(observed);sampled=np.asarray(sampled)
    if observed.shape!=field.shape or sampled.shape!=field.shape:raise ValueError('配准观测、背景采样与目标端口形状不一致')
    if not np.isfinite(observed).all() or not np.isfinite(sampled).all():raise ValueError('局部校正缺真实有限观测或背景支持')
    unit=field.attrs.get('units')
    if not unit:raise ValueError('局部端口单位未知')
    raw=RawInputQualityGate().validate(background,variable=variable,requires_spatial=True,expected_unit=unit)
    if raw['status']=='BLOCK':raise ValueError('背景质量门拒绝局部校正')
    meta=dict(units=unit,support='explicit native collocation only',temporal='exact timestamps')
    ports={k:Port(k,('units','support','temporal')) for k in ['native_cell','sampled_background','observation','residual','updated','validated_field']}
    def source(inputs,p,e):return dict(field=e[p['source']])
    def difference(inputs,p,e):return dict(field=Artifact('residual',residual_difference(inputs['observed'].value,inputs['sampled'].value),meta))
    def addition(inputs,p,e):return dict(field=Artifact('updated',residual_addition(inputs['background'].value,inputs['residual'].value),meta))
    checks={}
    def quality(inputs,p,e):
        out=background.copy(deep=True);out[variable].values[:]=inputs['field'].value
        checks['input']=raw
        checks['output']=ResultQualityGate().validate(background,out,variable=variable,semantic_type='continuous',expected_output_unit=unit)
        checks['raw_output']=RawInputQualityGate().validate(out,variable=variable,requires_spatial=True,expected_unit=unit)
        if any(v['status']=='BLOCK' for v in checks.values()):raise ValueError('局部校正结果未通过质量门')
        return dict(field=Artifact('validated_field',out,meta))
    adapters={
        ('local.background','run'):NodeAdapter({},dict(field=ports['native_cell']),source),
        ('local.sample','run'):NodeAdapter({},dict(field=ports['sampled_background']),source),
        ('local.observation','run'):NodeAdapter({},dict(field=ports['observation']),source),
        ('fusion.residual','difference'):NodeAdapter(dict(observed=ports['observation'],sampled=ports['sampled_background']),dict(field=ports['residual']),difference,mode='difference'),
        ('fusion.residual','add'):NodeAdapter(dict(background=ports['native_cell'],residual=ports['residual']),dict(field=ports['updated']),addition,mode='add'),
        ('local.quality','run'):NodeAdapter(dict(field=ports['updated']),dict(field=ports['validated_field']),quality)}
    from .plugins import ResidualFusionPlugin
    registry=AlgorithmRegistry();registry.register(ResidualFusionPlugin());task='collocated_residual_v1'
    registry.register_task_adapters(task,adapters,algorithm_ids=['fusion.residual'],
        boundary_ids=['local.background','local.sample','local.observation','local.quality'])
    nodes=[Node('background','local.background',parameters=dict(source='background',kind='native_cell')),
        Node('sample','local.sample',parameters=dict(source='sample',kind='sampled_background')),
        Node('observation','local.observation',parameters=dict(source='observation',kind='observation')),
        Node('residual','fusion.residual',dict(observed=('observation','field'),sampled=('sample','field')),dict(mode='difference')),
        Node('updated','fusion.residual',dict(background=('background','field'),residual=('residual','field')),dict(mode='add')),
        Node('quality','local.quality',dict(field=('updated','field')))]
    result=NumericalDAGExecutor(registry.task_adapters(task),version='1').run(nodes,'quality',external=dict(
        background=Artifact('native_cell',field.values.copy(),meta),sample=Artifact('sampled_background',sampled.copy(),meta),observation=Artifact('observation',observed.copy(),meta)))
    return result['outputs']['field'].value,dict(route=task,graph=[dict(id=n.id,operation=n.operation,inputs=n.inputs,parameters=n.parameters) for n in nodes],
        trace=result['trace'],quality=checks,accuracy_assessed=False,extrapolation=False,training=False,
        eligibility='same quantity/unit/exact time; observed point inside grid; finite native interpolation stencil',
        selection='single admissible local residual route; no accuracy ranking',
        assumption='local additive bias is transferred from the observation to its containing native cell only; not independent accuracy evidence')
