"""Public model-independent preparation entry point."""
from .general import EngineeringContextEngine
from .profiles import get_engineering_profile


def prepare_for_model(data, model, project_context=None):
    """Standardized data -> engineering mapping -> adapter-ready result."""
    return EngineeringContextEngine(get_engineering_profile(model)).prepare(data, project_context)
