"""Adapters are selected by Schema configuration, never by model name."""
from ecodata.core.exceptions import SchemaError


def get_adapter(schema):
    from ecodata.models.adapters.standardized import ModelAdapter
    from ecodata.models.adapters.cmaq import CMAQAdapter
    from ecodata.models.adapters.swat_plus import SWATPlusAdapter
    from ecodata.models.adapters.wasp import WASPAdapter
    from ecodata.models.adapters.modflow6_flopy import MODFLOW6FloPyAdapter
    adapters = {"standardized": ModelAdapter, "cmaq_auxiliary": CMAQAdapter,
                "swat_plus_text": SWATPlusAdapter,
                "modflow6_flopy": MODFLOW6FloPyAdapter,
                "wasp_package": ModelAdapter, "wasp_text": WASPAdapter}
    identifier = schema.preparation["adapter"]
    if identifier not in adapters:
        raise SchemaError(f"未注册Adapter：{identifier}；不能退回其他模型的处理规则")
    return adapters[identifier]()
