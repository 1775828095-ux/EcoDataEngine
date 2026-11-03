"""Small official-API smoke based on GOLEM synthetic graph-search quick start."""
from datetime import timedelta

from golem.core.adapter import IdentityAdapter
from golem.core.dag.verification_rules import has_no_cycle, has_no_self_cycled_nodes
from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.graph import OptGraph, OptNode
from golem.core.optimisers.objective import Objective, ObjectiveEvaluate
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams


def metric(graph):
    names = {node.name for node in graph.nodes}
    return len({"read", "process", "validate"} - names) + 0.01 * len(names)


objective = Objective({"coverage": metric})
optimizer = EvoGraphOptimizer(
    objective=objective,
    initial_graphs=[OptGraph([OptNode("read")])],
    requirements=GraphRequirements(num_of_generations=3, timeout=timedelta(seconds=20), show_progress=False),
    graph_generation_params=GraphGenerationParams(adapter=IdentityAdapter(),
        rules_for_constraint=[has_no_cycle, has_no_self_cycled_nodes],
        available_node_types=["read", "process", "validate"]),
    graph_optimizer_params=GPAlgorithmParameters(pop_size=6, max_pop_size=6),
)
result = optimizer.optimise(ObjectiveEvaluate(objective))[0]
print({"backend": "thegolem", "official_optimizer": type(optimizer).__name__,
       "result": [node.name for node in result.nodes], "fitness": metric(result)})
