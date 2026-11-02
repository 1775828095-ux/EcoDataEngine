from __future__ import annotations

from importlib.metadata import entry_points

from ecodata.algorithms.contracts import AlgorithmPlugin, AlgorithmSpec


class AlgorithmRegistry:
    """Registry for built-ins and third-party ``ecodata.algorithms`` entry points."""

    def __init__(self):
        self._plugins: dict[str, AlgorithmPlugin] = {}

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

