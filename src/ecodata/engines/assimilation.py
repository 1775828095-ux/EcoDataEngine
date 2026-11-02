"""Future DA contract inspired by JEDI/DART/DAPPER. No DA method implemented."""
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AssimilationContext:
    model_state: Any
    observations: Any
    observation_operator: Any
    variable_mapping: dict
    observation_error: Any
    background_error: Any
    method: str


class AssimilationEngine(Protocol):
    def can_run(self, context: AssimilationContext) -> dict: ...
    def prepare(self, context: AssimilationContext) -> AssimilationContext: ...
    def assimilate(self, context: AssimilationContext) -> Any: ...
    def validate(self, context: AssimilationContext, result: Any) -> dict: ...
