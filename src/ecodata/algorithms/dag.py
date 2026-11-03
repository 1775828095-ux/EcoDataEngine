"""Named-port numerical DAG execution in EcoData, independent of GOLEM internals."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import pickle
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr


class DAGError(ValueError):
    def __init__(self, message, trace=(), details=None):
        super().__init__(message)
        self.trace = list(trace)
        self.details = details or {}


@dataclass
class Artifact:
    kind: str
    value: Any
    metadata: dict = field(default_factory=dict)

    def fingerprint(self):
        digest = sha256()
        def text(value):
            digest.update(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))
        text([self.kind, self.metadata])
        if isinstance(getattr(self.value,"dataset",None),xr.Dataset):
            # Bundle pickle storage changes after a deepcopy/readback despite
            # equal scientific arrays. Use the existing dataset canonical hash.
            digest.update(Artifact(self.kind+".dataset",self.value.dataset).fingerprint().encode())
            from dataclasses import asdict
            text([{k:v for k,v in asdict(record).items() if k!="elapsed_seconds"} for record in self.value.history])
        elif isinstance(self.value, xr.Dataset):
            text(self.value.attrs)
            for name in sorted(self.value.variables):
                a = self.value[name]
                text([name, a.dims, str(a.dtype), a.shape, a.attrs])
                values = np.asarray(a.values)
                if values.dtype.kind == "O":
                    text(values.tolist())
                else:
                    digest.update(np.ascontiguousarray(values).tobytes())
        elif isinstance(self.value, pd.DataFrame):
            text([list(self.value.columns), [str(d) for d in self.value.dtypes], self.value.attrs])
            digest.update(pd.util.hash_pandas_object(self.value, index=True).values.tobytes())
        else:
            # Trained estimators may share a repr despite different fitted
            # weights. Cache state by content, never by their display string.
            digest.update(pickle.dumps(self.value, protocol=5))
        return digest.hexdigest()


@dataclass(frozen=True)
class Port:
    kind: str
    required_metadata: tuple[str, ...] = ()

    def validate(self, artifact):
        if artifact.kind != self.kind:
            raise DAGError(f"Port expects {self.kind}, received {artifact.kind}")
        missing = [name for name in self.required_metadata if artifact.metadata.get(name) is None or artifact.metadata.get(name) == ""]
        if missing:
            raise DAGError(f"Missing edge metadata: {missing}")


@dataclass
class Node:
    id: str
    operation: str
    inputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)


@dataclass
class NodeAdapter:
    inputs: dict[str, Port]
    outputs: dict[str, Port]
    execute: Callable
    # Modes are component execution contracts, not additional Registry candidates.
    mode: str = "run"
    version: str = "1"
    # Declarative output-path semantics; consumed by the shared selector, not
    # interpreted as a claim about numerical accuracy.
    decision_contract: dict = field(default_factory=dict)


class NumericalDAGExecutor:
    def __init__(self, adapters, *, version="1"):
        self.adapters = dict(adapters)
        self.version = version
        self.cache = {}

    def adapter(self, node):
        key = (node.operation, node.parameters.get("mode", "run"))
        if key not in self.adapters:
            raise DAGError(f"No executable adapter for {key}")
        return self.adapters[key]

    def order(self, nodes, sink):
        by_id = {n.id: n for n in nodes}
        if len(by_id) != len(nodes) or sink not in by_id:
            raise DAGError("Duplicate node id or missing sink")
        ordered, visiting, visited = [], set(), set()

        def visit(key):
            if key in visiting:
                raise DAGError("Cycle in numerical DAG")
            if key in visited:
                return
            if key not in by_id:
                raise DAGError(f"Missing parent {key}")
            visiting.add(key)
            n = by_id[key]
            adapter = self.adapter(n)
            if set(n.inputs) != set(adapter.inputs):
                raise DAGError(f"{key}: named inputs must be {sorted(adapter.inputs)}, got {sorted(n.inputs)}")
            for port, (parent, output) in n.inputs.items():
                visit(parent)
                upstream = self.adapter(by_id[parent])
                if output not in upstream.outputs or upstream.outputs[output].kind != adapter.inputs[port].kind:
                    raise DAGError(f"Incompatible edge {parent}.{output} -> {key}.{port}")
            visiting.remove(key)
            visited.add(key)
            ordered.append(n)

        visit(sink)
        if visited != set(by_id):
            raise DAGError("Disconnected/unconsumed nodes are not meaningful processing steps")
        return ordered

    def run(self, nodes, sink, *, external=None):
        values, trace = {}, []
        active_node = None
        try:
            ordered = self.order(nodes, sink)
            for node in ordered:
                active_node = node.id
                adapter = self.adapter(node)
                inputs, edges = {}, []
                for name, (parent, output) in node.inputs.items():
                    artifact = values[parent][output]
                    adapter.inputs[name].validate(artifact)
                    inputs[name] = artifact
                    edges.append({"from": f"{parent}.{output}", "to": f"{node.id}.{name}",
                                  "kind": artifact.kind, "fingerprint": artifact.fingerprint(), "status": "PASS"})
                # Source values, training data and state content are part of the key.
                source = (external or {}).get(node.parameters.get("source"))
                key = sha256(pickle.dumps((self.version, node.operation, adapter.version, node.parameters,
                    sorted((k, a.fingerprint()) for k, a in inputs.items()),
                    source.fingerprint() if source is not None else None), protocol=5)).hexdigest()
                started = perf_counter()
                hit = key in self.cache
                if hit:
                    outputs = deepcopy(self.cache[key])
                else:
                    outputs = adapter.execute(deepcopy(inputs), deepcopy(node.parameters), external or {})
                    if set(outputs) != set(adapter.outputs):
                        raise DAGError(f"{node.id}: incorrect output ports")
                    for name, artifact in outputs.items():
                        adapter.outputs[name].validate(artifact)
                    self.cache[key] = deepcopy(outputs)
                values[node.id] = outputs
                trace.append({"node": node.id, "operation": node.operation, "parameters": node.parameters,
                    "cache_key": key, "cache_hit": hit, "seconds": perf_counter()-started,
                    "edges": edges, "outputs": {k: {"kind": a.kind, "metadata": a.metadata,
                    "fingerprint": a.fingerprint()} for k, a in outputs.items()}, "status": "PASS"})
            return {"outputs": values[sink], "intermediates": values, "trace": trace}
        except Exception as exc:
            trace.append({"status": "FAIL", "node": active_node, "error": f"{type(exc).__name__}: {exc}"})
            raise DAGError(str(exc), trace, details=getattr(exc, "details", None)) from exc

    def from_golem(self, graph):
        """Resolve genuine GOLEM parent edges to uniquely typed named ports.

        Optional ``params._port_bindings`` map input names to [parent-index,
        output-port]. Bindings must participate in GOLEM's graph identity.
        Without bindings, inference is permitted only for a unique typed match.
        Ambiguous same-kind sources must never be ordered arbitrarily.
        """
        raw = list(graph.nodes)
        ids = {id(n): f"n{i}" for i, n in enumerate(raw)}
        nodes = [Node(ids[id(n)], n.name, parameters=dict(n.content.get("params", n.content.get("parameters", {})))) for n in raw]
        by_id = {n.id: n for n in nodes}
        consumed = set()
        for original, node in zip(raw, nodes):
            adapter = self.adapter(node)
            if "bindings" in original.content:
                raise DAGError("Store bindings in params._port_bindings so GOLEM includes them in graph identity")
            bindings = node.parameters.get("_port_bindings")
            if bindings is not None and set(bindings) != set(adapter.inputs):
                raise DAGError("Explicit bindings do not match named ports")
            used = set()
            for port, contract in adapter.inputs.items():
                if bindings is not None:
                    index, output = bindings[port]
                    if not isinstance(index, int) or not 0 <= index < len(original.nodes_from):
                        raise DAGError("Binding parent index out of range")
                    parent = original.nodes_from[index]
                    candidates = [(ids[id(parent)], output)]
                else:
                    candidates = [(ids[id(p)], name) for p in original.nodes_from
                        for name, c in self.adapter(by_id[ids[id(p)]]).outputs.items() if c.kind == contract.kind]
                if len(candidates) != 1:
                    raise DAGError(f"{node.operation}.{port}: missing or ambiguous parent binding")
                node.inputs[port] = candidates[0]
                used.add(candidates[0][0])
            parents = {ids[id(p)] for p in original.nodes_from}
            if used != parents:
                raise DAGError("Unused GOLEM parent edge")
            consumed |= parents
        sinks = set(by_id)-consumed
        if len(sinks) != 1:
            raise DAGError("Exactly one final output required")
        sink = sinks.pop()
        self.order(nodes, sink)
        return nodes, sink

    def run_golem(self, graph, *, external=None):
        nodes, sink = self.from_golem(graph)
        return self.run(nodes, sink, external=external)
