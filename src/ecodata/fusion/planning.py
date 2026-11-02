"""Role evidence and conservative prerequisites for multi-source fusion."""
from ecodata.recognition import inspect_dataset

ROLES = {"target_state", "observation", "forcing", "static_constraint", "initial_condition",
         "boundary_condition", "quality_auxiliary"}


class DataRoleClassifier:
    @staticmethod
    def physical_kind(dataset, variable):
        if variable not in dataset:
            return None
        names = {str(value).casefold().replace(" ", "") for value in
                 (variable, dataset[variable].attrs.get("standard_name", ""),
                  dataset[variable].attrs.get("physical_variable", ""))}
        return DataRoleClassifier._physical_names_kind(names)

    @staticmethod
    def _physical_names_kind(names):
        groups = {
            "terrain": {"elevation", "dem", "topography", "surface_altitude", "地形高程", "高程"},
            "meteorology": {"temperature", "air_temperature", "t2m", "t2", "wind", "wind_speed", "u10m", "v10m", "humidity", "relative_humidity", "rh", "气温", "温度", "风速", "湿度"},
            "pollution": {"pm25", "pm2.5", "mass_concentration_of_pm2p5_ambient_aerosol_particles_in_air", "no2", "so2", "o3", "co"},
        }
        for kind, aliases in groups.items():
            if names & aliases:
                return kind
        return None

    def compatibility(self, dataset, requirement, mapping=None, variables=()):
        """Physical evidence is a gate, never overridden by a user role label."""
        from ecodata.engines.units import UnitEngine
        role = str(requirement.get("role", requirement.get("name", ""))).casefold()
        expected = ("terrain" if role in {"terrain", "topography", "static_constraint"} else
                    "meteorology" if role in {"meteorology", "forcing"} else
                    "pollution" if role in {"background", "pollutant_observation", "initial_concentration", "boundary_concentration", "initial_concentrations", "boundary_concentrations"} else None)
        fields = requirement.get("fields", [])
        declared_kinds = {self._physical_names_kind({str(value).casefold().replace(" ", "") for value in
                          [field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])]}) for field in fields} - {None}
        if len(declared_kinds) == 1:
            expected = next(iter(declared_kinds))
        selected = list(variables) or [value for value in (mapping or {}).values() if value and not value.startswith("derived:")]
        selected = selected or list(dataset.data_vars)
        kinds = {self.physical_kind(dataset, name) for name in selected} - {None}
        reasons = []
        if expected and kinds and expected not in kinds:
            reasons.append(f"变量 {', '.join(selected)} 与 {role} 输入类型不兼容")
        for field in fields:
            source = (mapping or {}).get(field["name"])
            if not source or source not in dataset:
                continue
            target_kind = self._physical_names_kind({str(value).casefold().replace(" ", "") for value in
                [field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])]})
            source_kind = self.physical_kind(dataset, source)
            if source_kind and target_kind and source_kind != target_kind:
                reasons.append(f"变量 {source} 不能映射为 {field['name']}，物理语义不兼容")
        for source in selected:
            if self.physical_kind(dataset, source) == "terrain":
                unit = dataset[source].attrs.get("units", "")
                if unit:
                    try:
                        UnitEngine().affine(unit, "m")
                    except Exception as exc:
                        reasons.append(f"高程单位 {unit} 与长度不兼容：{exc}")
        if not fields and expected and expected not in kinds:
            reasons.append("无法可靠确认当前变量的物理用途，需要完善字段语义配置")
        return {"passed": not reasons, "status": "FAIL" if reasons else "PASS",
                "role": role, "physical_kinds": sorted(kinds), "reasons": reasons}

    def classify(self, dataset, variable):
        info = inspect_dataset("input.nc", dataset)
        kind = self.physical_kind(dataset, variable)
        if kind == "terrain" and not info.time_coordinate and info.x_coordinate and info.y_coordinate:
            return "static_constraint"
        if kind == "meteorology" and info.time_coordinate and info.structure != "table":
            return "forcing"
        explicit = dataset.attrs.get("data_role")
        if explicit:
            return explicit if explicit in ROLES else "unclassified"
        target_table = variable in dataset or ("value" in dataset and "variable" in dataset and (dataset["variable"] == variable).all().item())
        if target_table and info.structure == "table" and info.x_coordinate and info.y_coordinate:
            return "observation"
        if variable in dataset and info.x_coordinate and info.y_coordinate:
            return "target_state"
        if dataset.attrs.get("covariate_variables"):
            return "static_constraint" if not info.time_coordinate else "forcing"
        return "unclassified"


class FusionPlanner:
    def plan(self, datasets, variable, method="auto", covariates=()):
        roles = [DataRoleClassifier().classify(ds, variable) for ds in datasets]
        for i, ds in enumerate(datasets):
            if roles[i] == "unclassified" and any(name in ds for name in covariates):
                roles[i] = "forcing" if inspect_dataset("input.nc", ds).time_coordinate else "static_constraint"
        observations = [i for i, role in enumerate(roles) if role == "observation"]
        states = [i for i, role in enumerate(roles) if role in {"target_state", "initial_condition"}]
        auxiliaries = [i for i, role in enumerate(roles) if role in {"static_constraint", "forcing"}]
        templates = [i for i, ds in enumerate(datasets) if inspect_dataset("input.nc", ds).structure == "grid"
                     and inspect_dataset("input.nc", ds).x_coordinate and inspect_dataset("input.nc", ds).y_coordinate]
        target_grid = states[0] if states else templates[0] if templates else None
        reasons = []
        if method == "auto" and not covariates:
            covariates = tuple(name for ds in datasets for name in str(ds.attrs.get("covariate_variables", "")).split(",") if name)
        if method == "auto":
            method = "external_drift_kriging" if observations and covariates else "residual" if observations and states else "weighted" if len(states) > 1 else "ordinary_kriging"
        if method.endswith("kriging"):
            if len(observations) != 1 or target_grid is None:
                reasons.append("Kriging 需要一份显式训练站观测和一个带坐标/CRS的目标网格（可使用 DEM 网格，无须伪造背景）")
            if method == "external_drift_kriging" and not covariates:
                reasons.append("请选择具有明确物理含义的连续辅助字段，不能默认把所有数据相加")
            if method == "external_drift_kriging" and not auxiliaries and not any(name in ds for ds in datasets for name in covariates):
                reasons.append("没有可采样的辅助场")
        elif method == "weighted":
            from ecodata.fusion.alignment import fusion_compatibility
            if len(states) < 2:
                reasons.append("没有两份同变量网格场；辅助变量不能直接进入加权求和")
            else:
                reasons.extend(fusion_compatibility(datasets, variable)["reasons"])
        elif method == "residual" and (len(observations) != 1 or len(states) != 1):
            reasons.append("残差融合需要一份背景场与一份同变量观测；辅助场不能直接相加")
        return {"ready": not reasons, "method": method, "roles": roles, "observations": observations,
                "states": states, "auxiliaries": auxiliaries, "covariates": list(covariates), "reasons": reasons,
                "target_grid": target_grid,
                "relation_check": "执行前检查共同有效样本、覆盖、矩阵秩和训练留一关系；不足时拒绝而非强行融合"}
