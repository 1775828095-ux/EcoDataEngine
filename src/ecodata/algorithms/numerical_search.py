"""Shared result helpers; the retired GOLEM backend is intentionally unavailable."""
import hashlib
import json

RETIRED_MESSAGE = "旧GOLEM机制已移除，请使用当前平台的PortraitMutation随机变异入口。"

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

def record(graph):
    nodes=list(graph.nodes)
    return [dict(operation=n.name,parameters=dict(n.parameters),parents=[nodes.index(p) for p in n.nodes_from]) for n in nodes]


def key_for(graph,factory):return hashlib.sha256(factory.inspect(graph)["key"].encode()).hexdigest()


def run_numerical_search(*args, **kwargs):
    raise ValueError(RETIRED_MESSAGE)


def restore(*args, **kwargs):
    raise ValueError(RETIRED_MESSAGE)
