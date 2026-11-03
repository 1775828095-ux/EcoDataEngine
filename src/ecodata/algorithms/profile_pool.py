"""Task-local frozen scientific pool over existing adapters and typed grammar."""
import hashlib
import inspect
from pathlib import Path


def freeze_pool(factory, registry, value, max_steps, fingerprint):
    if value is None:
        return dict(status="UNFROZEN_LEGACY_POOL", fingerprint="unfrozen")
    if not isinstance(value, dict) or set(value) != {"algorithms"}:
        raise ValueError("candidate_pool requires exactly algorithms")
    requested=value["algorithms"]
    if not isinstance(requested,list) or not 1<=len(requested)<=64 or any(not isinstance(x,str) for x in requested):
        raise ValueError("candidate_pool requires 1..64 registered algorithm IDs")
    from ecodata.algorithms.raster_reconstruction_task import RasterBaselineAdapter
    from ecodata.algorithms.plugins import RegridPlugin
    from ecodata.algorithms.open_source_spatial import XarrayAggregationPlugin
    registered={s.id for s in registry.specs()}
    aliases={};canonical=set()
    for identifier in requested:
        if identifier not in registered:raise ValueError("Unknown frozen algorithm: "+identifier)
        plugin=registry.get(identifier)
        if plugin.spec.status!="available":raise ValueError("Frozen algorithm is not available: "+identifier)
        # Explicit frozen raster pools choose one common boundary convention.
        # This is not a claim that a raw interpolator and its boundary wrapper
        # have identical behaviour outside the source-centre envelope.
        base=plugin.upstream if isinstance(plugin,RasterBaselineAdapter) else plugin
        wrapper="raster_task."+base.spec.id
        reviewed=(isinstance(base,RegridPlugin) and base.method in {"nearest","bilinear"}) or (
            isinstance(base,XarrayAggregationPlugin) and base.method=="cubic")
        wrapped_mode=any(a.decision_contract.get("algorithm_id",key[0])==wrapper for key,a in factory.adapters.items())
        target=wrapper if reviewed and wrapper in registered and wrapped_mode else identifier
        aliases[identifier]=target;canonical.add(target)
    specifications=[]
    for identifier in sorted(canonical):
        plugin=registry.get(identifier)
        implementations={Path(inspect.getfile(type(plugin)))}
        if isinstance(plugin,RasterBaselineAdapter):implementations.add(Path(inspect.getfile(type(plugin.upstream))))
        for key,adapter in factory.adapters.items():
            if adapter.decision_contract.get("algorithm_id",key[0])==identifier:
                implementations.add(Path(inspect.getfile(adapter.execute)))
        specifications.append(dict(spec=plugin.spec.as_dict(), implementation_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(implementations)}))
    adapters={}
    for key,adapter in factory.adapters.items():
        identifier=adapter.decision_contract.get("algorithm_id",key[0])
        if key[0].startswith(("source.","quality.")) or adapter.decision_contract.get("structural_only") or identifier in canonical:
            adapters[key]=adapter
    available={a.decision_contract.get("algorithm_id",key[0]) for key,a in adapters.items()}
    missing=canonical-available
    if missing:raise ValueError("Frozen canonical method has no executable task adapter: "+", ".join(sorted(missing)))
    factory.adapters=adapters;factory.producers={};factory._memo.clear()
    for key,adapter in adapters.items():
        for output,port in adapter.outputs.items():factory.producers.setdefault(port.kind,[]).append((key,output))
    identity=dict(version="frozen_scientific_pool_v1",algorithms=sorted(canonical),max_steps=max_steps,
                  boundary_policy="reviewed raster wrappers use explicit nearest edge extension; PCHIP clips coordinates",
                  specifications=specifications)
    return dict(identity,status="FROZEN",requested_algorithms=requested,aliases=aliases,fingerprint=fingerprint(identity),
                deduplication="same scientific kernel and explicit common boundary policy; not output-value deduplication")
