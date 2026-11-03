from __future__ import annotations

from importlib.metadata import entry_points

from ecodata.algorithms.contracts import AlgorithmPlugin, AlgorithmSpec


class AlgorithmRegistry:
    """Registry for built-ins and third-party ``ecodata.algorithms`` entry points."""

    def __init__(self):
        self._plugins: dict[str, AlgorithmPlugin] = {}
        self._task_adapters: dict[str, dict] = {}

    def register(self, plugin: AlgorithmPlugin, *, replace: bool = False) -> AlgorithmPlugin:
        algorithm_id = plugin.spec.id
        if algorithm_id in self._plugins and not replace:
            raise ValueError(f"Algorithm already registered: {algorithm_id}")
        self._plugins[algorithm_id] = plugin
        return plugin

    def get(self, algorithm_id: str) -> AlgorithmPlugin:
        try:
            return self._plugins[algorithm_id]
        except KeyError as exc:
            raise KeyError(f"Unknown algorithm: {algorithm_id}") from exc

    def specs(self) -> tuple[AlgorithmSpec, ...]:
        return tuple(plugin.spec for _, plugin in sorted(self._plugins.items()))

    def register_task_adapters(self, task_id: str, adapters: dict, *, algorithm_ids, boundary_ids=()) -> None:
        """Bind execution modes to existing plugins without adding algorithm IDs."""
        from ecodata.algorithms.dag import NodeAdapter
        allowed, boundaries = set(algorithm_ids), set(boundary_ids)
        for (operation, mode), adapter in adapters.items():
            if operation not in boundaries:
                if operation not in allowed:
                    raise ValueError(f"Task adapter is outside the frozen pool: {operation}")
                self.get(operation)  # A JSON name alone is not a registered plugin.
            if not isinstance(adapter, NodeAdapter) or not mode or not adapter.outputs:
                raise ValueError(f"Invalid named-port adapter: {operation}/{mode}")
        self._task_adapters[task_id] = dict(adapters)

    def task_adapters(self, task_id: str) -> dict:
        if task_id not in self._task_adapters:
            raise KeyError(f"No executable task adapters registered: {task_id}")
        return dict(self._task_adapters[task_id])

    def load_entry_points(self) -> tuple[str, ...]:
        loaded = []
        for item in entry_points(group="ecodata.algorithms"):
            plugin = item.load()()
            self.register(plugin)
            loaded.append(plugin.spec.id)
        return tuple(loaded)


def create_default_registry() -> AlgorithmRegistry:
    from ecodata.algorithms.plugins import builtin_plugins

    registry = AlgorithmRegistry()
    for plugin in builtin_plugins():
        registry.register(plugin)
    return registry

