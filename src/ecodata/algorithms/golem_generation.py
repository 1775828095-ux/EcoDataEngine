"""Import compatibility for historical task definitions; no GOLEM backend."""
from ecodata.algorithms.dag import DAGError
from ecodata.algorithms.numerical_search import RETIRED_MESSAGE

class GenerationError(DAGError):
    pass

class TypedGolemFactory:
    def __init__(self, *args, **kwargs):
        raise ValueError(RETIRED_MESSAGE)

def typed_tree_growth(*args, **kwargs):
    raise ValueError(RETIRED_MESSAGE)
