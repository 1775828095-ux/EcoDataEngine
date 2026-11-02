"""Build ordered, auditable plans from matched requirements."""
from ecodata.core.exceptions import SchemaError


class ProcessingPlanner:
    def __init__(self, schema, adapter):
        self.schema, self.adapter = schema, adapter

    def build(self, inputs, assignments, matches):
        remaining = {role["role"]: role for role in self.schema.inputs}
        ordered = []
        while remaining:
            ready = [name for name, role in remaining.items() if set(role["depends_on"]) <= set(ordered)]
            if not ready:
                raise SchemaError("模型输入依赖存在循环，无法生成处理顺序")
            for name in ready:
                ordered.append(name)
                remaining.pop(name)
        errors, steps = [], []
        by_role = {item["role"]: item for item in matches}
        roles = {role["role"]: role for role in self.schema.inputs}
        for name in ordered:
            match = by_role[name]
            unmet = [dep for dep in roles[name]["depends_on"] if not by_role[dep]["can_process"]]
            if unmet and not roles[name].get("record_only"):
                match.setdefault("reasons", []).append("未满足输入依赖：" + ", ".join(unmet))
                match.update(can_process=False, status="待确认/当前无法处理")
        for match in matches:
            if match["required"] and not match["can_process"]:
                errors.append(match["name"] + "：" + "；".join(match.get("missing", []) + match.get("reasons", [])))
            if match["can_process"]:
                for operation in match.get("operations", []):
                    steps.append({"stage": match["name"], **operation})
        plan = {"model": self.schema.profile["name"], "version": self.schema.profile["version"],
                "scenario": self.schema.preparation["id"], "adapter": self.schema.preparation["adapter"],
                "scope": "standardized_preparation", "assignments": assignments, "matches": matches,
                "errors": errors, "steps": steps, "execution_order": ordered, "compatibility": {},
                "can_process": not errors, "can_process_partial": self.adapter.supports_partial and any(m["can_process"] for m in matches),
                "native_model_input": False, "complete_model_input": False,
                "native_gaps": self.schema.preparation.get("native_gaps", []),
                "unassigned_files": sorted({i.filename for i in inputs} - set(assignments.values()))}
        self.adapter.validate_plan(self.schema, inputs, assignments, plan)
        plan["role_conflicts"] = [match for match in matches if match.get("role_compatibility", {}).get("passed") is False]
        if plan["role_conflicts"]:
            plan["errors"].extend("角色不兼容：" + "；".join(match["role_compatibility"]["reasons"]) for match in plan["role_conflicts"])
            plan["can_process_partial"] = False
        plan["can_process"] = not plan["errors"]
        return plan
