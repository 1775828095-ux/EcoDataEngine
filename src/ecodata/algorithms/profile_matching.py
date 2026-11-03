"""Evidence-bearing portraits and bounded short-plan selection, without evolution.

Existing contracts, typed recipes, numerical kernels and quality boundaries remain
authoritative. A model portrait is a scoped preparation contract, not proof that
the native simulator can read the generated product.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import hashlib
import json
import re

import numpy as np

from ecodata.algorithms.capabilities import describe_algorithm
from ecodata.algorithms.semantic_requirements import SemanticFeatureExtractor
from ecodata.models.requirements import RequirementStore, GENERIC_UNITS


MODELS = ("calpuff", "wasp", "swat", "swat_plus", "wrf", "camx", "cmaq", "wrf_chem")
VERSION = "portraits_short_plans_v3_reference_free"
MODEL_ROOT = Path(__file__).resolve().parents[3] / "output/plan_ranking_models"


class RecommendationOnly:
    protocol = dict(kind='recommendation_only', objective='NOT_ASSESSED', accuracy_claim=False)

    def __call__(self, dataset):
        raise ValueError('Recommendation-only mode must not execute or fabricate an evaluation')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def model_table(model_id):
    if model_id not in MODELS:
        raise ValueError("Model is outside the reviewed profile entries")
    profile = RequirementStore().get(model_id)
    scenario = profile.get("preparation_scenario", {})
    roles = []
    for item in scenario.get("inputs", profile["inputs"]):
        roles.append(dict(role=item.get("role", item["name"]), required=item.get("required", True),
                          fields=item.get("fields", []), time=item.get("time"),
                          space=item.get("space"), record_only=item.get("record_only", False)))
    return dict(model_id=model_id, version=profile["version"], roles=roles,
                sources=profile["reference_sources"], profile_sha256=fingerprint(profile),
                status=profile["status"], native_acceptance="NOT_ASSESSED",
                scope="reviewed field-level preparation only; no full-model sensitivity claim",
                sensitivity=None, sensitivity_status="UNKNOWN")


def prepare_model_request(request):
    """Resolve a reviewed field. Never infer a simulator's needs from its name."""
    if 'application_contract' in request:
        from ecodata.algorithms.application_contract import prepare
        return prepare(request)
    spec = request.get("model_profile", {})
    if set(spec) != {"id", "role", "field"}:
        raise ValueError("model_profile requires exactly id, role and field")
    if 'converter_context' in request:
        from ecodata.models.preparation_bridge import resolve_target
        table = resolve_target(spec['id'], request['converter_context'])
    else:
        table = model_table(spec["id"])
    role = next((r for r in table["roles"] if r["role"] == spec["role"]), None)
    if role is None or role["record_only"]:
        raise ValueError("No executable reviewed field contract for this model role")
    field = next((f for f in role["fields"] if f["name"] == spec["field"]), None)
    if field is None:
        raise ValueError("No reviewed field definition; do not invent a model requirement")
    unit = field.get("unit", "")
    if unit.casefold() in GENERIC_UNITS:
        raise ValueError("Model field unit is UNKNOWN; confirm the contract first")
    if request.get("unit") != unit:
        raise ValueError("Model field unit differs; explicit unit conversion is required before this task")
    kind = field.get("data_type", field.get("kind", "continuous"))
    if request.get("semantic_type") != kind:
        raise ValueError("Model field type conflicts with the requested variable semantics")
    result = dict(request)
    from ecodata.algorithms.attribute_registry import configuration
    registry,scoring_parameters=configuration(request)
    from ecodata.algorithms.portrait_rules import model_requirements
    state_policy=None
    requirements=request.get("model_requirements", {})
    if request.get("portrait_policy"):
        from ecodata.models.portrait_policy import resolve
        state_policy=resolve(request,table,registry)
        requirements=dict(requirements,priorities={i["attribute"]:i["importance"] for i in state_policy["scenario"]["issues"]},
            evidence=state_policy["scenario"]["evidence"])
    matching = model_requirements(requirements,registry)
    definitions={a['id']:a for a in registry['attributes']}
    if any(definitions[k]['property']=='coarse_consistency' for k in matching["hard"]):
        result["requirements"] = dict(request.get("requirements", {}), require_coarse_consistency=True)
    # Only machine-readable, reviewed time requirements are promoted. Free text
    # such as 'simulation-specific' is never guessed to mean daily or hourly.
    if 'converter_context' in request:
        interval = table['converter_contract'].get('coordinates', {}).get('time', {}).get('interval_seconds')
        temporal = {3600:'1h',86400:'1D'}.get(interval)
    else:
        temporal = next((r.get("temporal_resolution") for r in RequirementStore().get(spec["id"])["inputs"]
                         if r["name"] == spec["role"]), None)
    if temporal in {"1h", "1D"} and request["task"] == "temporal":
        parameters = dict(request.get("parameters", {}))
        if parameters.get("target_time_frequency", temporal) != temporal:
            raise ValueError("Target frequency conflicts with the reviewed model field")
        parameters["target_time_frequency"] = temporal
        result["parameters"] = parameters
    table.update(selected_role=spec["role"], selected_field=field,
                 selected_requirement_status="CONFIRMED_PREPARATION_CONTRACT",
                 full_model_ready=False,
                 remaining_roles=[r['role'] for r in table['roles'] if r['required'] and r['role']!=spec['role']],
                 matching_requirements=matching,attribute_registry=registry,scoring_parameters=scoring_parameters,
                 application_preferences=validate_preferences(request.get("preferences", {})))
    if state_policy is not None:table["state_policy"]=state_policy
    return result, table


def validate_preferences(value):
    if set(value) - {"weights", "scales", "thresholds", "tie_tolerance"}:
        raise ValueError("Unknown preference has no score consumer")
    weights = value.get("weights", {"objective": 1.0})
    scales = value.get("scales", {k: 1.0 for k in weights})
    thresholds = value.get("thresholds", {})
    allowed = {"objective", "RMSE", "MAE", "abs_bias"}
    if not weights or not set(weights) <= allowed or set(scales) != set(weights) or not set(thresholds) <= allowed:
        raise ValueError("Unsupported metric or missing normalization scale")
    for name, values in (("weight", weights), ("scale", scales), ("threshold", thresholds)):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
               or (v <= 0 if name == "scale" else v < 0) for v in values.values()):
            raise ValueError("Invalid " + name)
    if not sum(weights.values()) > 0:
        raise ValueError("At least one positive preference weight required")
    tolerance = value.get("tie_tolerance", 0.0)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Invalid tie tolerance")
    return dict(weights=weights, scales=scales, thresholds=thresholds, tie_tolerance=tolerance)


def task_loss(row, preferences):
    metrics = dict(row, abs_bias=abs(row["Bias"]) if row.get("Bias") is not None else None)
    for key in set(preferences["weights"]) | set(preferences["thresholds"]):
        if not isinstance(metrics.get(key), (int, float)) or not np.isfinite(metrics[key]):
            raise ValueError("Required objective component is unavailable: " + key)
    if any(metrics[k] > bound for k, bound in preferences["thresholds"].items()):
        return None
    total = sum(preferences["weights"].values())
    return float(sum(w / total * metrics[k] / preferences["scales"][k]
                     for k, w in preferences["weights"].items()))


def portraits(context, registry, model):
    from ecodata.algorithms.dag import Artifact
    from ecodata.algorithms.portrait_rules import data_facts, algorithm_properties
    data = []
    for index, source in enumerate(context.sources):
        semantics = SemanticFeatureExtractor().extract(source.dataset, context.variable)
        field = source.dataset[context.variable]
        numeric = np.asarray(field.values)
        shape_features = {}
        if numeric.ndim == 2 and context.target_lat is not None:
            shape_features = SemanticFeatureExtractor.raster_features(source.dataset, context.variable, len(context.target_lat))
        data.append(dict(source_index=index, semantics=semantics.as_dict(),
                         content_hash=Artifact('portrait_input',source.dataset[[context.variable]]).fingerprint(),
                         matching_facts=data_facts(context, index),
                         visible_features=shape_features, feature_version="existing_pixel_index_v1",
                         morphology_status="CONFIRMED" if shape_features else "NOT_APPLICABLE_OR_MISSING",
                         gradient_units="normalized pixel-index descriptor; not physical-distance derivative",
                         noise=None, dynamics=None, processing_history=[str(x) for x in source.history],
                         available_at="visible algorithm inputs only"))
    if context.observations is not None or context.auxiliaries:
        data[0]["source_field_hash"]=data[0]["content_hash"]
        data[0]["content_hash"]=Artifact("portrait_visible_inputs",dict(
            sources=[s.dataset for s in context.sources],auxiliaries=context.auxiliaries,
            observations=context.observations)).fingerprint()
    if model.get("state_policy"):
        from ecodata.algorithms.portrait_state import scientific_identity
        data[0]["scientific_content_hash"]=scientific_identity(context)
    algorithms = []
    for spec in registry.specs():
        algorithms.append(dict(asdict(spec), effective_capabilities=list(describe_algorithm(spec).capabilities),
                               declared_cost=describe_algorithm(spec).compute_cost,
                               matching_properties=algorithm_properties(registry.get(spec.id)),
                               empirical_side_effects=None, empirical_cost=None,
                               evidence_status="Use declared evidence; not revalidated by portrait construction"))
    task_context=dict(semantic_type=context.semantic_type,target_lat=context.target_lat,target_lon=context.target_lon,
        target_time_frequency=context.parameters.get("target_time_frequency"),output_unit=context.model_constraints.get("output_unit"),
        source_grids=[dict(crs=s.dataset.attrs.get("crs"),
            coordinates={axis:s.dataset[axis].values.tolist() for axis in ("lat","lon") if axis in s.dataset.coords}) for s in context.sources])
    if model.get("state_policy"):
        task_context["state_measurement_sha256"]=hashlib.sha256(Path(__file__).with_name("portrait_state.py").read_bytes()).hexdigest()
        task_context["state_policy_context"]=dict(version=model["state_policy"]["policy_version"],
            sha256=model["state_policy"]["policy_sha256"],scenario=model["state_policy"]["scenario"],memory_version=model["state_policy"].get("memory_version"))
    if context.parameters.get("feature_groups"):
        task_context["feature_prediction_protocol"]={key:context.parameters.get(key) for key in
            ("feature_groups","gp_length_scale_m","gp_signal_std","gp_noise_std","include_station_only_baselines")}
    return dict(version=VERSION, data=data, algorithms=algorithms, model=model, task_context=task_context)


def validate_model_context(context, model):
    field = model["selected_field"]
    names = {field["name"], field.get("standard_name", field["name"]), *field.get("aliases", [])}
    for source in context.sources:
        actual = source.dataset[context.variable]
        identity = actual.attrs.get("standard_name", actual.attrs.get("physical_variable", context.variable))
        if identity not in names:
            raise ValueError("Input physical variable does not match the selected model field")


def attach_stage_portraits(factory, variable, context=None):
    """Instrument actual output artifacts, including cached metadata, in this mode only."""
    from dataclasses import replace
    import xarray as xr
    for key, adapter in list(factory.adapters.items()):
        if key[0].startswith(("source.", "quality.")):
            continue
        def execute(inputs, parameters, external, original=adapter.execute):
            outputs = original(inputs, parameters, external)
            for artifact in outputs.values():
                value = artifact.value
                ds = value if isinstance(value, xr.Dataset) else getattr(value, "dataset", None)
                if ds is None and hasattr(value, "data"):
                    ds = getattr(value.data, "dataset", None)
                if isinstance(ds, xr.Dataset) and variable in ds:
                    facts = SemanticFeatureExtractor().extract(ds, variable).as_dict()
                    artifact.metadata = dict(artifact.metadata, stage_portrait=facts,
                                             portrait_available_at="actual upstream computation, no evaluation labels")
                    if context is not None:
                        from ecodata.algorithms.portrait_rules import data_facts
                        from ecodata.core.schemas import EcoDataBundle
                        artifact.metadata["stage_matching_facts"] = data_facts(replace(context,sources=(EcoDataBundle(ds),)),0)
            return outputs
        factory.adapters[key] = replace(adapter, execute=execute)


def public_vector(tables, steps, protocol):
    """Whitelist observed descriptors. No reference, filename or observed loss."""
    model = tables["model"]
    features = dict(model=model["model_id"], model_version=model["version"],
                    model_field=model["selected_field"]["name"], unit=model["selected_field"]["unit"],
                    pool=tables.get("candidate_pool",{}).get("fingerprint","unfrozen"),
                    task_context=fingerprint(tables["task_context"]),
                    length=len(steps))
    for index, data in enumerate(tables["data"]):
        features.update({f"source{index}.{k}":v for k,v in data["visible_features"].items()})
        observed = data["semantics"]["observed"]["value"]
        features.update({f"source{index}.{k}":observed[k] for k in ("missing_fraction", "finite_count") if k in observed})
        for key, value in data["matching_facts"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                features[f"source{index}.matching.{key}"] = value
    for index, step in enumerate(steps):
        features["step" + str(index)] = step
        algorithm = next(a for a in tables["algorithms"] if a["id"] == step)
        features["transfer" + str(index)] = algorithm["matching_properties"]["transfer"]
        for capability in algorithm["effective_capabilities"]:
            features[f"step{index}.capability.{capability}"] = 1
        features[f"step{index}.declared_cost"] = algorithm["declared_cost"]
    return features


def objective_semantics(protocol):
    """Label-head identity only, never a physical-metric predictor feature."""
    if protocol.get("kind") == "recommendation_only":return None
    return fingerprint(dict(kind=protocol.get("kind"),objective=protocol.get("objective",protocol.get("metric"))))


def ranking_context(features):
    return fingerprint({k:features[k] for k in ('model','model_version','model_field','unit','pool','task_context')})


class PlanRankingService:
    """Legacy diagnostic regressors; never replace the joint scoring policy."""
    @classmethod
    def publish_policy(cls,model_id,published_id,*,learning_order,fallback_policy,validation):
        import shutil
        if learning_order not in {'metric','task_rank'} or fallback_policy not in {'rules','development_fixed'}:
            raise ValueError('Invalid published policy')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',published_id):raise ValueError('Invalid published model ID')
        original=cls(model_id);destination=MODEL_ROOT/published_id
        if destination.exists():raise ValueError('Immutable published model already exists')
        if len(original.report.get('rank_preferences_sha256',[]))!=1:
            raise ValueError('Policy requires unambiguous training preferences')
        if fallback_policy=='development_fixed' and 'development_fixed' not in original.payload:
            raise ValueError('Policy has no development baseline')
        destination.mkdir(parents=True)
        for name in ['model.joblib','portrait_evidence.json']:shutil.copyfile(MODEL_ROOT/model_id/name,destination/name)
        report=dict(original.report,model_id=published_id,parent_model_id=model_id,
            default_policy=dict(learning_order=learning_order,fallback_policy=fallback_policy),
            policy_validation=dict(path=str(validation),sha256=fingerprint_bytes(validation)))
        (destination/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        return report

    @classmethod
    def train(cls, folders, model_id, *, holdout_groups):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_id):
            raise ValueError("Invalid ranking model ID")
        destination = MODEL_ROOT / model_id
        if destination.exists():
            raise ValueError("Immutable ranking model already exists")
        records = []; seen = set(); held = set(holdout_groups); source_groups = {}
        for folder in folders:
            report = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
            if report.get("profile_version") != VERSION or report.get("evidence_role") != "development" or not report.get("group"):
                raise ValueError("Only explicit grouped development runs are ranking evidence")
            if report.get("stop_reason") != "CANDIDATE_POOL_COMPLETED":
                raise ValueError("Incomplete candidate coverage cannot provide complete rank labels")
            for source in report['portraits']['data']:
                identity=source['content_hash']
                if identity in source_groups and source_groups[identity] != report['group']:
                    raise ValueError("Same source content cannot cross development groups")
                source_groups[identity]=report['group']
            valid = sorted((r for r in report["rows"] if r.get("status") == "PASS" and r.get("task_loss") is not None), key=lambda r:r["task_loss"])
            if len(valid) < 2:
                continue
            identity = fingerprint(dict(data=report["portraits"]["data"], evaluation=report["evaluation_protocol"],
                                        candidates=[r["key"] for r in valid]))
            if identity in seen:
                continue
            seen.add(identity)
            losses = np.array([r["task_loss"] for r in valid])
            from scipy.stats import rankdata
            labels = 1 - (rankdata(losses, method="average") - 1) / (len(valid) - 1)
            for row, label in zip(valid, labels):
                if row.get("file_readback") != "PASS" or not row.get("numerical_execution_attempted"):
                    raise ValueError("Actual execution and product readback required")
                product = Path(row["product"])
                if not product.is_file() or fingerprint_bytes(product) != row.get("product_sha256"):
                    raise ValueError("Ranking evidence product fingerprint mismatch")
                # Reconcile labels with the immutable numerical executor record,
                # rather than trusting editable summary metrics alone.
                executed = json.loads((product.parent / "result.json").read_text(encoding="utf-8"))
                for key in ("objective", "RMSE", "MAE", "Bias"):
                    if row.get(key) != executed.get(key):
                        raise ValueError("Ranking label differs from numerical execution record: " + key)
                if task_loss(row, report["portraits"]["model"]["application_preferences"]) != row["task_loss"]:
                    raise ValueError("Ranking task loss does not match recorded preferences")
                metrics = {k: row[k] for k in ("objective", "RMSE", "MAE") if isinstance(row.get(k), (int, float)) and np.isfinite(row[k])}
                if row.get("Bias") is not None:metrics["abs_bias"] = abs(row["Bias"])
                records.append(dict(features=row["public_features"], label=float(label), group=report["group"], metrics=metrics,
                                    objective_semantics=objective_semantics(report["evaluation_protocol"]),
                                    row=row, portraits=report["portraits"],
                                    combination_gain=next((g for g in report.get("combination_gains",[]) if g["key"]==row["key"]),None),
                                    summary=str(Path(folder)/"summary.json"), summary_sha256=fingerprint_bytes(Path(folder)/"summary.json")))
        train = [r for r in records if r["group"] not in held]
        test = [r for r in records if r["group"] in held]
        if len({r["group"] for r in train}) < 2 or not test or not held <= {r["group"] for r in test}:
            raise ValueError("Need two development training groups and disjoint holdout groups")
        objective_heads={r["objective_semantics"] for r in train}
        if len(objective_heads)!=1 or None in objective_heads:
            raise ValueError("Different objective semantics cannot share a training model; RMSE and coverage are not interchangeable")
        vectorizer = DictVectorizer(sparse=False)
        x = vectorizer.fit_transform([r["features"] for r in train])
        estimator = RandomForestRegressor(n_estimators=64, max_depth=4, min_samples_leaf=2, random_state=17, n_jobs=1)
        estimator.fit(x, [r["label"] for r in train])
        support = [r["features"] for r in train]
        metric_models = {}
        for metric in sorted(set.intersection(*(set(r["metrics"]) for r in train))):
            regressor = RandomForestRegressor(n_estimators=64, max_depth=4, min_samples_leaf=2, random_state=17, n_jobs=1)
            regressor.fit(x, [r["metrics"][metric] for r in train])
            metric_models[metric] = regressor
        payload = dict(vectorizer=vectorizer, estimator=estimator, support=support, metric_models=metric_models,
                       objective_semantics=next(iter(objective_heads)))
        fixed_losses={}
        for item in train:
            key=(ranking_context(item['features']),tuple(item['row']['steps']))
            fixed_losses.setdefault(key,[]).append(item['row']['task_loss'])
        payload['development_fixed']={}
        for (context_key,steps),losses in fixed_losses.items():
            payload['development_fixed'].setdefault(context_key,[]).append(dict(steps=list(steps),
                mean_training_loss=float(np.mean(losses)),cases=len(losses)))
        destination.mkdir(parents=True)
        path = destination / "model.joblib"; joblib.dump(payload, path)
        evidence = dict(version=VERSION, scope="training groups only; observations are context-specific, not universal algorithm/model properties",
                        algorithms={}, data_cases=[], model_cases=[], combination_cases=[])
        for item in train:
            row=item["row"]; source=dict(summary=item["summary"], summary_sha256=item["summary_sha256"],
                product=row["product"], product_sha256=row["product_sha256"], key=row["key"], group=item["group"])
            case=dict(source=source, steps=row["steps"], metrics=item["metrics"],
                      matching={k:row.get("matching",{}).get(k) for k in ("score","score_interval","properties")},
                      task_loss=row["task_loss"], costs=row.get("costs", {}))
            evidence["data_cases"].append(dict(case, data=item["portraits"]["data"]))
            evidence["model_cases"].append(dict(case, model={k:v for k,v in item["portraits"]["model"].items()
                                                          if k not in {"empirical_preparation_cases", "learning_scope"}}))
            if len(row["steps"])>1:
                evidence["combination_cases"].append(dict(case, observed_gain=item["combination_gain"]))
            if len(row["steps"]) == 1:
                import xarray as xr
                with xr.open_dataset(row["product"]) as output:
                    variable=item["portraits"]["data"][0]["semantics"]["variable"]
                    values=np.asarray(output[variable].values)
                    finite=values[np.isfinite(values)]
                    observation=dict(case, finite_fraction=float(np.isfinite(values).mean()),
                        minimum=float(finite.min()) if finite.size else None, maximum=float(finite.max()) if finite.size else None,
                        negative_fraction=float((finite<0).mean()) if finite.size else None,
                        cost_scope="end-to-end execution/quality includes cache; not isolated kernel timing")
                evidence["algorithms"].setdefault(row["steps"][0], []).append(observation)
        evidence_path=destination/"portrait_evidence.json"
        evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        report = dict(model_id=model_id, status="LIMITED_DEVELOPMENT", version=VERSION,
                      training_groups=sorted({r["group"] for r in train}), holdout_groups=sorted(held),
                      training_rows=len(train), holdout_rows=len(test), model_sha256=fingerprint_bytes(path),
                      metric_models=sorted(metric_models), portrait_evidence_sha256=fingerprint_bytes(evidence_path),
                      objective_semantics=payload["objective_semantics"],
                      evidence="Development labels only; rank prediction is not accuracy or calibrated confidence")
        report['rank_preferences_sha256']=sorted({fingerprint(r['portraits']['model']['application_preferences']) for r in train})
        from importlib.metadata import version
        report['scikit_learn_version']=version('scikit-learn')
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        service = cls(model_id)
        predictions = [service.score(r["features"]) for r in test]
        usable = [(p, r) for p, r in zip(predictions, test) if p is not None]
        report.update(holdout_supported=len(usable), holdout_mae=float(np.mean([abs(p-r["label"]) for p,r in usable])) if usable else None)
        metric_validation={}
        for metric in metric_models:
            errors=[]
            for item in test:
                prediction=service.predict_metrics(item["features"], objective_head=item["objective_semantics"])
                if prediction is not None and metric in prediction and metric in item["metrics"]:
                    errors.append(abs(prediction[metric]-item["metrics"][metric]))
            metric_validation[metric]=dict(supported=len(errors), mae=float(np.mean(errors)) if errors else None)
        report["metric_validation"]=metric_validation
        (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def __init__(self, model_id):
        import joblib
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_id):raise ValueError("Invalid ranking model ID")
        root = MODEL_ROOT / model_id
        self.report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        if self.report.get("version") != VERSION:raise ValueError("Ranking portrait schema version mismatch; retraining required")
        from importlib.metadata import version
        if self.report.get('scikit_learn_version')!=version('scikit-learn'):
            raise ValueError('Ranking runtime version mismatch')
        if fingerprint_bytes(root / "model.joblib") != self.report["model_sha256"]:
            raise ValueError("Ranking model fingerprint mismatch")
        self.payload = joblib.load(root / "model.joblib")
        evidence_path=root/"portrait_evidence.json"
        if fingerprint_bytes(evidence_path) != self.report["portrait_evidence_sha256"]:
            raise ValueError("Portrait evidence fingerprint mismatch")
        self.portrait_evidence=json.loads(evidence_path.read_text(encoding="utf-8"))

    def score(self, features):
        if not self.supports(features):return None
        return float(self.payload["estimator"].predict(self.payload["vectorizer"].transform([features]))[0])

    def supports(self, features):
        return self.support_report(features)['supported']

    def support_report(self, features):
        support = self.payload["support"]
        # Categorical context AND ordered plan must already have been seen.
        categories = {k:v for k,v in features.items() if isinstance(v, str)}
        matching = [r for r in support if {k:v for k,v in r.items() if isinstance(v,str)} == categories and set(r) == set(features)]
        if not matching:return dict(supported=False,reason='UNSEEN_CONTEXT_OR_SCHEMA',matching_rows=0,violations=[])
        violations=[]
        for k,v in features.items():
            if isinstance(v,str):continue
            lower=min(r[k] for r in matching);upper=max(r[k] for r in matching)
            if not np.isfinite(v) or not lower<=v<=upper:
                violations.append(dict(feature=k,value=v,lower=lower,upper=upper))
        return dict(supported=not violations,reason='OUTSIDE_NUMERIC_SUPPORT' if violations else 'SUPPORTED',
                    matching_rows=len(matching),violations=violations)

    def predict_metrics(self, features, *, objective_head=None):
        if not self.supports(features):return None
        x=self.payload["vectorizer"].transform([features])
        return {key:float(model.predict(x)[0]) for key,model in self.payload["metric_models"].items()
                if key!="objective" or objective_head==self.payload["objective_semantics"]}


def fingerprint_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_profile_selection(evaluator, config, folder, request, model, context, *, seed=11, progress=None):
    from golem.core.optimisers.graph import OptGraph
    from ecodata.algorithms.numerical_search import save, key_for, record
    from ecodata.algorithms.portrait_rules import score_graph, validate_product
    started = perf_counter(); folder = Path(folder); factory = evaluator.factory
    options = request.get("profile_options", {})
    if set(options) - {"max_steps", "candidate_limit", "candidate_pool", "ranking_model_id", "acknowledge_limited_development", "evidence_role", "group", "learning_order", "fallback_policy", "search"}:
        raise ValueError("Unknown profile option")
    learning_order=options.get('learning_order','metric')
    if learning_order not in {'metric','task_rank'}:raise ValueError('Unknown learning_order')
    fallback_policy=options.get('fallback_policy','rules')
    if fallback_policy not in {'rules','development_fixed'}:raise ValueError('Unknown fallback_policy')
    max_steps = options.get("max_steps", 3); limit = options.get("candidate_limit", 64)
    if type(max_steps) is not int or not 1 <= max_steps <= 6 or type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("Bounded plans require 1..6 scientific steps and 1..128 recipe cap; default remains 3")
    for key, maximum in (("objective_limit",200),("wall_seconds",600)):
        if not isinstance(config[key],(int,float)) or not 0 < config[key] <= maximum:raise ValueError("Invalid profile budget: "+key)
    if options.get("evidence_role", "evaluation") not in {"development", "evaluation"}:raise ValueError("Invalid evidence role")
    tables = portraits(context, evaluator.pipeline.registry, model)
    from ecodata.algorithms.profile_pool import freeze_pool
    pool=freeze_pool(factory,evaluator.pipeline.registry,options.get("candidate_pool"),max_steps,fingerprint)
    tables["candidate_pool"]=pool
    calibration_id=request.get('scoring',{}).get('calibration_id')
    if calibration_id:
        from ecodata.algorithms.scoring_calibration import resolve
        tables['model']['scoring_parameters'],tables['model']['calibration']=resolve(calibration_id,tables)
    save(folder/"candidate_pool.json",pool)
    service = None
    if options.get("ranking_model_id"):
        if not options.get("acknowledge_limited_development"):raise ValueError("Explicit limited-development acknowledgement required")
        service = PlanRankingService(options["ranking_model_id"])
        defaults=service.report.get('default_policy',{})
        learning_order=options.get('learning_order',defaults.get('learning_order','metric'))
        fallback_policy=options.get('fallback_policy',defaults.get('fallback_policy','rules'))
        if learning_order not in {'metric','task_rank'} or fallback_policy not in {'rules','development_fixed'}:
            raise ValueError('Invalid model default policy')
        # Full evidence stays immutable beside the model. Bounded excerpts avoid
        # copying every development product into every public recommendation.
        def excerpt(items):
            return items if len(items)<=16 else [items[i] for i in np.linspace(0,len(items)-1,16,dtype=int)]
        tables['history_index']=dict(path=str(MODEL_ROOT/options['ranking_model_id']/'portrait_evidence.json'),
            sha256=service.report['portrait_evidence_sha256'],display='at most 16 evenly spaced examples per list; full training evidence retained',
            counts={k:len(service.portrait_evidence[k]) for k in ('data_cases','model_cases','combination_cases')})
        for algorithm in tables["algorithms"]:
            observations=excerpt(service.portrait_evidence["algorithms"].get(algorithm["id"], []))
            algorithm["empirical_observations"]=observations
            algorithm["empirical_side_effects"] = [dict(minimum=o["minimum"], maximum=o["maximum"],
                negative_fraction=o["negative_fraction"], finite_fraction=o["finite_fraction"], source=o["source"]) for o in observations] or None
            algorithm["empirical_cost"] = [dict(costs=o["costs"],scope=o["cost_scope"],source=o["source"]) for o in observations] or None
        tables["historical_data_cases"]=excerpt(service.portrait_evidence["data_cases"])
        tables["historical_combinations"]=excerpt(service.portrait_evidence["combination_cases"])
        tables["model"]["empirical_preparation_cases"]=excerpt(service.portrait_evidence["model_cases"])
        tables["model"]["learning_scope"]="field preparation outcomes only; simulator sensitivity remains UNKNOWN"
    attach_stage_portraits(factory, context.variable, context)
    if model.get("state_policy"):
        from ecodata.algorithms.portrait_state import instrument
        instrument(factory,context)
    # The executor uses the registry's adapter copy; update this task only.
    evaluator.pipeline.registry.register_task_adapters(evaluator.task_id, factory.adapters,
        algorithm_ids=[s.id for s in evaluator.pipeline.registry.specs()],
        boundary_ids=[key[0] for key in factory.adapters if key[0].startswith(("source.","prepare.","quality."))])
    preferences = model["application_preferences"]
    if evaluator.evaluation_protocol["kind"] == "declared_task_metric" and any(k != "objective" for k in set(preferences["weights"]) | set(preferences["thresholds"])):
        raise ValueError("Accuracy preference requires an independent reference")
    factory.local_recipe_seed = seed; factory.reverse_limit = limit; factory._memo.clear()
    recipes = factory.recipes("validated_field")
    if pool["status"]=="FROZEN" and (len(recipes)>=limit or getattr(factory,"collaboration_truncated",False)):
        raise ValueError("Frozen pool may be truncated; increase candidate_limit before execution")
    candidates = []; rejected = []; seen = set()
    registered = {s.id for s in evaluator.pipeline.registry.specs()}
    for _, recipe in recipes:
        graph = OptGraph(factory.instantiate(recipe)); key = key_for(graph, factory)
        if key in seen:continue
        seen.add(key)
        steps = []
        for node in reversed(list(graph.nodes)):
            identifier = factory.adapter(node).decision_contract.get('algorithm_id', node.name)
            if identifier in registered and identifier != 'validation.result':steps.append(identifier)
        if not 1 <= len(steps) <= max_steps:
            rejected.append(dict(key=key, reason="scientific step limit", steps=steps));continue
        try:evaluator.decision().require_graph(factory, graph)
        except Exception as exc:rejected.append(dict(key=key,reason=str(exc)));continue
        matching = score_graph(tables, graph, factory)
        if matching["hard_status"] != "PASS":
            rejected.append(dict(key=key, reason="model portrait hard requirement", steps=steps,
                                 graph=record(graph), matching=matching));continue
        vector = public_vector(tables, steps, evaluator.evaluation_protocol)
        vector['structure'] = fingerprint(record(graph))
        candidates.append(dict(key=key, graph=graph, steps=steps, public_features=vector, matching=matching,
                               declared_cost=sum(describe_algorithm(evaluator.pipeline.registry.get(s).spec).compute_cost for s in steps), predicted_rank=None))
    probe_report=None
    definitions=tables["model"]["attribute_registry"]["attributes"]
    if any(a["provider"]=="station_probe" and a["scoring_enabled"] and
           tables["model"]["matching_requirements"]["priorities"].get(a["id"],0)>0 for a in definitions):
        from ecodata.algorithms.station_probe import measure_candidates
        probe_report=measure_candidates(evaluator,context,request,candidates,folder)
        tables["station_probes"]=probe_report["measurements"]
        tables["data"][0]["available_station_portrait"]=dict(count=len(context.observations),
            value_std=float(context.observations.value.std(ddof=0)),
            station_ids=context.observations.station_id.tolist(), source="available fitting observations only")
        for item in candidates:
            item["matching"]=score_graph(tables,item["graph"],factory)
    state_evolution=None
    if model.get("state_policy"):
        if probe_report:raise ValueError("State matching and full-pool probe scoring are separate policies; do not blend silently")
        from ecodata.algorithms.portrait_state import score as score_state
        from ecodata.algorithms.portrait_memory import load as load_memory
        memory_version=model["state_policy"].get("memory_version")
        memory=load_memory(memory_version,tables) if memory_version else None
        tables["state_memory_status"]=dict(version=memory_version,loaded=memory is not None,affects="conditional state responses, never task ranking")
        for item in candidates:item["matching"]=score_state(tables,item["graph"],factory,context,memory)
        search_options=options.get("search")
        if search_options:
            from ecodata.algorithms.portrait_evolution import search
            state_evolution=search(candidates,factory,evaluator.decision(),search_options,folder,seed=seed,wall_seconds=max(.1,config["wall_seconds"]-(perf_counter()-started)))
            if not state_evolution["exact_check"]:
                considered=set(state_evolution["visited"])
                candidates=[c for c in candidates if c["key"] in considered]
    elif options.get("search"):
        raise ValueError("Portrait genetic search requires the explicit three-portrait state policy")
    for item in candidates:
        item['support_diagnostics']=service.support_report(item['public_features']) if service else None
        item["predicted_metrics"] = service.predict_metrics(item["public_features"],objective_head=objective_semantics(evaluator.evaluation_protocol)) if service else None
        item["predicted_task_loss"] = None
        predicted=item["predicted_metrics"]
        if predicted is not None and set(preferences["weights"]) <= set(predicted):
            item["predicted_task_loss"] = sum(w/sum(preferences["weights"].values()) * predicted[k]/preferences["scales"][k]
                                               for k,w in preferences["weights"].items())
            item["predicted_rank"] = service.score(item["public_features"]) if service.report.get('rank_preferences_sha256')==[fingerprint(preferences)] else None
    pool.update(grammar_candidate_count=len(recipes),legal_candidate_count=len(candidates),
                generation_complete_within_cap=len(recipes)<limit)
    save(folder/"candidate_pool.json",pool)
    # Do not silently mix an estimated rank with absent evidence on another scale.
    learned = bool(candidates) and all(c["predicted_task_loss"] is not None for c in candidates)
    learning_status=dict(model_loaded=service is not None,supported_candidates=sum(c["predicted_metrics"] is not None for c in candidates),
        target_supported_candidates=sum(c["predicted_task_loss"] is not None for c in candidates),requested_metrics=sorted(preferences["weights"]),
        learning_order=learning_order,
        fallback_policy=fallback_policy,development_fixed_used=False,
        diagnostic_predictions_supported=learned,affects_selection=False,
        ordering="three-portrait joint suitability lower bound; legacy model predictions are diagnostics only")
    rule_active = bool(model["matching_requirements"]["priorities"])
    def ordering(c):
        return (-c["matching"]["score_interval"][0] if rule_active else 0,
                c["declared_cost"], len(c["steps"]), c["key"])
    candidates.sort(key=ordering)
    save(folder / "portraits.json", tables)
    ranking_status = "THREE_PORTRAIT_STATE_REDUCTION" if model.get("state_policy") else "THREE_PORTRAIT_JOINT_SCORE" if rule_active else "RULE_COST_ORDER_NOT_ACCURACY"
    if model.get("state_policy") and not model["state_policy"]["active_issue_count"]:
        ranking_status="NO_ACTIVE_STATE_TERM_COST_ORDER"
    recommendation_certainty="NOT_APPLICABLE"
    if model.get("state_policy") and candidates:
        best_bound=candidates[0]["matching"]["score_interval"][0]
        recommendation_certainty="ROBUST_UNDER_RECORDED_BOUNDS" if all(best_bound>=c["matching"]["score_interval"][1] for c in candidates[1:]) else "UNRESOLVED_REQUIRES_PROBE"
    save(folder / "candidate_ranking.json", dict(status=ranking_status,
         candidates=[dict(c,graph=record(c["graph"])) for c in candidates], rejected=rejected,
         generation_cap=limit, exhaustive=False, model=service.report if service else None,
         candidate_pool=pool,reference_values_in_predictor=False,
         learning_status=learning_status,
         rule_order="descending lower suitability bound; then declared cost, step count, key; NOT accuracy confidence",
         unknown_candidates=sum(c["matching"]["score"] is None for c in candidates),
         learned_order="diagnostics only; cannot override joint scoring or trigger a fixed-model fallback"))
    if probe_report:
        save(folder / "probe_costs.json", {k:v for k,v in probe_report.items() if k!="measurements"})
    rows=[]; stop="CANDIDATE_POOL_COMPLETED" if candidates else "NO_FEASIBLE_CANDIDATES"
    if evaluator.evaluation_protocol['kind']=='recommendation_only':
        return dict(profile_version=VERSION,mode='Profile',stop_reason='RECOMMENDATION_ONLY',rows=[],best=None,
                    recommendations=[dict(c,graph=record(c['graph'])) for c in candidates],portraits=tables,
                    evaluation_protocol=evaluator.evaluation_protocol,matching_status=ranking_status,
                    candidate_pool=pool,actual_accuracy="NOT_ASSESSED",
                    learning_status=learning_status,
                    scoring_policy="joint_state_v1" if model.get("state_policy") else "joint_portrait_v1",
                    state_evolution=state_evolution,recommendation_certainty=recommendation_certainty,recommended_key=candidates[0]['key'] if candidates else None,
                    accuracy_threshold_status="UNVERIFIED_NO_REFERENCE",
                    counts=dict(real_executions=0,candidate_count=len(candidates),rejected=len(rejected),
                                training_probe_graph_folds=probe_report["graph_fold_attempts"] if probe_report else 0),
                    selection_basis=ranking_status+'; structural suitability only; accuracy NOT_ASSESSED',
                    evolution_used=state_evolution is not None,native_model_acceptance='NOT_ASSESSED',seconds=perf_counter()-started)
    for item in candidates:
        if len(rows) >= config["objective_limit"]:stop="EVALUATION_LIMIT";break
        if perf_counter()-started >= config["wall_seconds"]:stop="WALL_TIME";break
        row=evaluator.execute(item["graph"],item["key"])
        row.update(public_features=item["public_features"],steps=item["steps"],predicted_rank=item["predicted_rank"],
                   predicted_metrics=item["predicted_metrics"], predicted_task_loss=item["predicted_task_loss"],
                   matching=item["matching"], task_loss=None)
        if row["status"] == "PASS":
            import xarray as xr
            with xr.open_dataset(row["product"]) as product:
                row["model_product_acceptance"] = validate_product(product, context.variable, tables)
            if model.get("state_policy"):
                from ecodata.models.portrait_policy import validate_coupled
                with xr.open_dataset(row["product"]) as product:
                    coupled=validate_coupled(product,model["state_policy"])
                row["model_product_acceptance"]["coupled"]=coupled
                if coupled["status"]=="FAIL":row["model_product_acceptance"]["status"]="FAIL"
            if row["model_product_acceptance"]["status"] == "FAIL":
                row.update(status="FAIL", error="Model portrait numerical hard requirement failed")
            else:
                row["task_loss"] = task_loss(row, preferences)
            row["product_sha256"] = fingerprint_bytes(row["product"])
            row["preference_status"] = "PASS" if row["task_loss"] is not None else "THRESHOLD_REJECTED"
        rows.append(row)
        if progress:progress(dict(evaluated=len(rows),status=row["status"]))
    valid=[r for r in rows if r["status"]=="PASS" and r["task_loss"] is not None]
    best=valid[0] if valid else None
    hindsight_best=None
    if valid:
        minimum=min(r["task_loss"] for r in valid)
        near=[r for r in valid if r["task_loss"]<=minimum+preferences["tie_tolerance"]]
        costs={c["key"]:c["declared_cost"] for c in candidates}
        hindsight_best=min(near,key=lambda r:(-r["matching"]["score_interval"][0] if rule_active else 0,
                                   costs[r["key"]],len(r["steps"]),r["task_loss"],r["key"]))
    from ecodata.algorithms.portrait_rules import combination_gains
    gains = combination_gains(rows, evaluator.evaluation_protocol)
    return dict(profile_version=VERSION,mode="Profile",stop_reason=stop,rows=rows,best=best,
                scoring_policy="joint_state_v1" if model.get("state_policy") else "joint_portrait_v1",
                    state_evolution=state_evolution,recommendation_certainty=recommendation_certainty,recommended_key=candidates[0]['key'] if candidates else None,
                hindsight_best=hindsight_best,hindsight_scope="observed executed-candidate minimum; diagnostic, never the recommendation policy",
                evaluation_protocol=evaluator.evaluation_protocol,portraits=tables,
                evidence_role=options.get("evidence_role","evaluation"),group=options.get("group"),
                candidate_pool=pool,
                learning_status=learning_status,
                counts=dict(real_executions=len(rows),training_probe_graph_folds=probe_report["graph_fold_attempts"] if probe_report else 0,quality_failures=sum(r["status"]!="PASS" for r in rows),
                            candidate_count=len(candidates),rejected=len(rejected),node_cache_hits=evaluator.node_hits),
                combination_gains=gains, matching_status=ranking_status,
                learning_record_status="ELIGIBLE_GROUPED_DEVELOPMENT" if options.get("evidence_role")=="development" and options.get("group") and stop=="CANDIDATE_POOL_COMPLETED" else "RECORDED_NOT_TRAINING_ELIGIBLE",
                selection_basis="joint portrait score chooses recommendation and first passing delivery; actual losses are evaluation/calibration evidence",evolution_used=state_evolution is not None,
                native_model_acceptance="NOT_ASSESSED",seconds=perf_counter()-started)
