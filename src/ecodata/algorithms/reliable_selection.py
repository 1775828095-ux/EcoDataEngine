"""Shared measured multi-fold loss and global, practical-equivalence selection.

No graph execution, hidden reference access or learned ranker lives here.  The
caller applies scientific hard gates and supplies every required fold's real
metrics.  Unmeasured attributes remain unknown rather than receiving a score.
"""
from pathlib import Path
from hashlib import sha256
import json
import math

import numpy as np


VERSION = 'reliable_selection_v2_measured_incumbent'
POLICY = Path(__file__).with_name('reliable_selection_policy.json')
METRICS = ('RMSE', 'MAE', 'abs_bias')
MODES = ('full', 'fold_only', 'no_portraits', 'no_parsimony', 'legacy_single')


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Finite numeric ' + label + ' required')
    if value <= 0 if positive else value < 0:
        raise ValueError(('Positive ' if positive else 'Nonnegative ') + label + ' required')
    return float(value)


def _weights(value):
    if not isinstance(value, dict) or not value or not set(value) <= set(METRICS):
        raise ValueError('Explicit RMSE/MAE/abs_bias weights required')
    weights = {name: _number(weight, name + ' weight') for name, weight in value.items()}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError('At least one positive measured-loss weight required')
    return {name: weight / total for name, weight in sorted(weights.items()) if weight > 0}


def _published(request, calibration):
    if calibration is None:
        if not POLICY.exists():
            return {}, dict(status='UNCALIBRATED_EXACT_LOSS_ONLY', source=None)
        content = POLICY.read_bytes()
        calibration = json.loads(content)
        source = dict(path=str(POLICY), sha256=sha256(content).hexdigest())
    else:
        calibration = dict(calibration)
        source = dict(explicit=True, sha256=_digest(calibration))
    if calibration.get('evidence_role') != 'development' or not calibration.get('evidence'):
        raise ValueError('Published practical tolerance requires development evidence')
    scope = calibration.get('scope', {})
    if any(request.get(name) != value for name, value in scope.items()):
        return {}, dict(status='OUT_OF_CALIBRATION_SCOPE_EXACT_LOSS_ONLY', source=source, scope=scope)
    return calibration, dict(status='DEVELOPMENT_CALIBRATED', source=source, scope=scope,
                             evidence=calibration['evidence'], version=calibration.get('version'))


def make_policy(request, retained_values, *, model_weights=None, calibration=None):
    """Freeze a task loss using visible values and explicit model priorities.

    ``request['reliable_selection']`` accepts mode, delta, scales and evidence.
    A positive explicit delta needs a stated basis.  Published defaults must
    carry development provenance; an absent/inapplicable publication uses zero
    practical tolerance, never an invented calibrated threshold.
    """
    declaration = request.get('reliable_selection', {})
    if not isinstance(declaration, dict):
        raise ValueError('Structured reliable_selection declaration required')
    allowed = {'mode', 'delta', 'scales', 'weights', 'evidence'}
    if set(declaration) - allowed:
        raise ValueError('Unknown reliable-selection option')
    mode = declaration.get('mode', 'full')
    if mode not in MODES:
        raise ValueError('Unknown reliable-selection mode')
    values = np.asarray(retained_values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        raise ValueError('Finite retained inputs required; no reference-derived scale')
    variation = float(np.std(finite))
    scale = variation if variation > 0 else 1.
    scale_basis = ('retained visible input population standard deviation' if variation > 0 else
                   'unit-scale engineering fallback for constant retained input; not empirical uncertainty')
    preferences = request.get('preferences', {})
    supplied = model_weights if model_weights is not None else preferences.get('weights', {'objective': 1.})
    if not isinstance(supplied, dict):
        raise ValueError('Structured model metric weights required')
    if 'objective' in supplied:
        if set(supplied) != {'objective'}:
            raise ValueError('Generic objective cannot be mixed with explicit metric semantics')
        _number(supplied['objective'], 'generic objective weight', positive=True)
        weights = {'RMSE': 1.}
        weight_basis = 'Default declared reconstruction RMSE; generic legacy objective was not treated as a measured metric'
    else:
        weights = _weights(supplied)
        weight_basis = 'Explicit application/model metric priorities'
    if 'weights' in declaration:
        declared = _weights(declaration['weights'])
        if 'objective' not in supplied and declared != weights:
            raise ValueError('Reliable policy cannot override explicit model metric priorities')
        weights = declared
        weight_basis = 'Explicit reliable-selection metric priorities'
    portraits = mode not in {'fold_only', 'no_portraits', 'legacy_single'}
    if not portraits:
        weights = {'RMSE': 1.}
        weight_basis = 'Explicit no-portrait ablation: measured RMSE only; hard gates unchanged'
    provided_scales = declaration.get('scales', preferences.get('scales', {}))
    if not isinstance(provided_scales, dict):
        raise ValueError('Structured metric normalization scales required')
    if set(provided_scales) - set(METRICS) - {'objective'}:
        raise ValueError('Unknown metric normalization scale')
    scales = {name: _number(provided_scales[name], name + ' scale', positive=True)
              if name in provided_scales else scale for name in weights}
    published, publication = _published(request, calibration)
    if published.get('metric_weights',weights)!=weights:
        published={}
        publication=dict(publication,status='OUT_OF_CALIBRATION_OBJECTIVE_EXACT_LOSS_ONLY')
    delta = _number(declaration.get('delta', published.get('delta', 0.)), 'practical loss tolerance')
    if delta > 0 and 'delta' in declaration and not declaration.get('evidence'):
        raise ValueError('Explicit positive tolerance requires development/engineering evidence')
    parsimony = mode in {'full', 'no_portraits'}
    roundoff = float(np.finfo(float).eps * max(1, len(finite)) * max(1., float(abs(finite).max())) /
                     min(scales.values()))
    policy = dict(version=VERSION, mode=mode, portraits_enabled=portraits, parsimony_enabled=parsimony,
        weights=weights, scales=scales, delta=delta, roundoff_tolerance=roundoff,
        scale_basis={name: 'Explicit application/task scale' if name in provided_scales else scale_basis for name in weights},
        weight_basis=weight_basis, tolerance=publication, explicit_evidence=declaration.get('evidence'),
        common_reference_graph=published.get('common_reference_graph'),
        data_portrait=dict(retained_finite_count=int(len(finite)), total_cells=int(values.size),
            retained_fraction=float(len(finite) / values.size), retained_standard_deviation=variation),
        aggregation='Equal required-fold mean; all folds mandatory; folds are not independent statistical replicates',
        objective_semantics='Weighted normalized RMSE/MAE/absolute-bias losses; lower is better')
    policy['sha256'] = _digest(policy)
    return policy


def _metric(metrics, name):
    if name == 'abs_bias' and metrics.get('Bias') is not None:
        signed = metrics['Bias']
        if isinstance(signed, bool) or not isinstance(signed, (int, float)) or not math.isfinite(signed):
            raise ValueError('Finite measured Bias required')
        value = abs(signed)
    else:value = metrics.get(name)
    if value is None:
        raise ValueError('Required measured metric is UNKNOWN: ' + name)
    return _number(value, 'measured ' + name)


def aggregate(fold_rows, policy, expected_folds):
    """Evaluate one graph on an identical, complete required-fold collection.

    Fold rows contain ``fold_id``, ``status='PASS'`` and a ``metrics`` mapping.
    ``objective_fold_ids`` optionally selects the primary objective support.
    Other folds remain mandatory diagnostics and never silently enter the mean.
    ``objective`` is normalized task loss; physical metrics stay in their units.
    """
    expected = list(expected_folds)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Unique nonempty required fold IDs needed')
    declared = policy.get('objective_fold_ids', expected)
    if (not isinstance(declared, (list, tuple)) or not declared or
            len(set(declared)) != len(declared) or not set(declared) <= set(expected)):
        raise ValueError('Unique nonempty objective-fold subset of required folds needed')
    primary = [identifier for identifier in expected if identifier in set(declared)]
    diagnostic = [identifier for identifier in expected if identifier not in set(declared)]
    by_id = {}
    for row in fold_rows:
        if row.get('fold_id') in by_id:
            raise ValueError('Duplicate fold response')
        by_id[row.get('fold_id')] = row
    if set(by_id) != set(expected):
        raise ValueError('Incomplete or mismatched required folds; no successful-subset score')
    losses = []; terms = []; raw = []; valid = []
    for identifier in expected:
        row = by_id[identifier]
        if row.get('status') not in {'PASS', 'ELIGIBLE_OUTPUT'}:
            raise ValueError('Required fold did not pass: ' + str(identifier))
        metrics = row.get('metrics')
        if not isinstance(metrics, dict):
            raise ValueError('Real per-fold metrics required')
        contributions = []
        for name, weight in policy['weights'].items():
            value = _metric(metrics, name)
            normalized = value / policy['scales'][name]
            contributions.append(dict(metric=name, data_activation=1.,
                data_evidence='Declared reconstruction objective, evaluated on this required fold',
                algorithm_response=value, response_status='MEASURED_COMPLETE_GRAPH',
                model_weight=weight, model_evidence=policy['weight_basis'],
                normalization_scale=policy['scales'][name], scale_evidence=policy['scale_basis'][name],
                contribution=weight * normalized))
        loss = float(sum(term['contribution'] for term in contributions))
        losses.append(loss); terms.append(dict(fold_id=identifier, loss=loss, terms=contributions,
            role='objective' if identifier in primary else 'diagnostic'))
        raw.append(dict(metrics))
        n = metrics.get('denominator', metrics.get('valid_n'))
        if n is not None:
            n = _number(n, 'reference support', positive=True)
            if not n.is_integer():raise ValueError('Integer reference support required')
        valid.append(n)
    primary_indices = [index for index, identifier in enumerate(expected) if identifier in primary]
    diagnostic_indices = [index for index, identifier in enumerate(expected) if identifier in diagnostic]
    def statistics(indices):
        if not indices:return None
        values = [losses[index] for index in indices]
        return dict(mean=float(np.mean(values)), std=float(np.std(values)),
            minimum=float(min(values)), maximum=float(max(values)),
            std_interpretation='Dispersion across these folds; NOT standard error or confidence interval')
    def support(indices):
        return int(sum(valid[index] for index in indices)) if all(valid[index] is not None for index in indices) else None
    definition = dict(weights=policy['weights'], scales=policy['scales'],
        aggregation='Equal objective-fold mean; every required fold must pass', objective_fold_ids=primary)
    result = dict(objective=float(np.mean([losses[index] for index in primary_indices])),
        fold_ids=expected, fold_losses=losses, objective_fold_ids=primary, diagnostic_fold_ids=diagnostic,
        loss_stats=statistics(primary_indices), diagnostic_loss_stats=statistics(diagnostic_indices),
        valid_n=support(primary_indices), all_folds_valid_n=support(range(len(expected))),
        diagnostic_valid_n=support(diagnostic_indices),
        fold_metrics=raw, loss_definition=definition, loss_signature=_digest(definition),
        portrait_response=dict(status='MEASURED_MULTI_FOLD', terms=terms,
            data=policy['data_portrait'], model_weight_basis=policy['weight_basis'],
            objective_fold_ids=primary, diagnostic_fold_ids=diagnostic,
            unknown_model_sensitivity='No unmeasured simulator response is inferred',
            predicted_gain='UNKNOWN until a comparable common baseline is measured'),
        complete_required_folds=True, fold_count=len(expected), objective_fold_count=len(primary),
        folds_are_independent=False)
    for name in ('RMSE', 'MAE', 'Bias'):
        values = [raw[index].get(name) for index in primary_indices]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values):
            result[name] = float(np.mean(values))
        else:result[name] = None
    return result


def _complexity(row):
    count = row.get('algorithm_nodes')
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError('Nonnegative effective scientific node count required')
    # Observed wall time and median runtime estimates deliberately never break ties.
    cost = row.get('cost_units', row.get('declared_cost'))
    if cost is None:cost = math.inf
    else:cost = _number(cost, 'declared deterministic cost')
    return int(count), cost, str(row['key'])


def ordered(rows, policy):
    """One global equivalence set; no non-transitive pairwise tolerance ranks."""
    eligible = [row for row in rows if row.get('status', 'ELIGIBLE_OUTPUT') in {'PASS', 'ELIGIBLE_OUTPUT'} and not row.get('dominated')]
    if not eligible:return []
    loss = lambda row: _number(row.get('metrics', {}).get('objective'), 'candidate objective')
    minimum = min(map(loss, eligible))
    tolerance = max(policy['roundoff_tolerance'], policy['delta'] if policy['parsimony_enabled'] else 0.)
    near = [row for row in eligible if loss(row) <= minimum + tolerance]
    rest = [row for row in eligible if loss(row) > minimum + tolerance]
    if policy['parsimony_enabled']:
        near.sort(key=_complexity)
        rest.sort(key=lambda row: (loss(row), *_complexity(row)))
    else:
        near.sort(key=lambda row: str(row['key']))
        rest.sort(key=lambda row: (loss(row), str(row['key'])))
    return near + rest


def choose(rows, policy):
    ranked = ordered(rows, policy)
    return ranked[0] if ranked else None


def protected_order(rows,policy,baseline,*,support_sufficient):
    """Rank complete comparable measured evidence, including a better new graph.

    Representativeness limits claims; it does not invalidate an isolated real
    fold or require every fold to improve. Hard gates remain in the evaluator.
    The existing global equivalence/simplicity rule decides numerical ties.
    """
    rows=[row for row in rows if row.get('metrics',{}).get('complete_required_folds') is True]
    if baseline is not None:
        metrics=baseline['metrics']
        rows=[row for row in rows if row['metrics'].get('fold_ids')==metrics.get('fold_ids')
              and row['metrics'].get('loss_signature')==metrics.get('loss_signature')]
    return ordered(rows,policy)


def common_gain(candidate_metrics, baseline_metrics, policy):
    """Common-reference gains are diagnostics, never a second accuracy reward."""
    if candidate_metrics.get('fold_ids') != baseline_metrics.get('fold_ids'):
        raise ValueError('Common-reference gain requires identical folds')
    if candidate_metrics.get('loss_signature') != baseline_metrics.get('loss_signature'):
        raise ValueError('Common-reference gain requires identical metric semantics and scales')
    gains = [b - a for a, b in zip(candidate_metrics['fold_losses'], baseline_metrics['fold_losses'])]
    primary = candidate_metrics.get('objective_fold_ids', candidate_metrics['fold_ids'])
    diagnostic = candidate_metrics.get('diagnostic_fold_ids', [])
    objective_gains = [gain for identifier, gain in zip(candidate_metrics['fold_ids'], gains) if identifier in primary]
    diagnostic_gains = [gain for identifier, gain in zip(candidate_metrics['fold_ids'], gains) if identifier in diagnostic]
    value = float(baseline_metrics['objective'] - candidate_metrics['objective'])
    baseline = baseline_metrics['objective']
    return dict(status='MEASURED_COMMON_REFERENCE', gain=value, fold_gains=gains,
        fold_ids=candidate_metrics['fold_ids'], objective_fold_ids=primary, diagnostic_fold_ids=diagnostic,
        objective_fold_gains=objective_gains, diagnostic_fold_gains=diagnostic_gains,
        fold_wins=sum(v > policy['roundoff_tolerance'] for v in objective_gains),
        diagnostic_fold_wins=sum(v > policy['roundoff_tolerance'] for v in diagnostic_gains),
        all_fold_wins=sum(v > policy['roundoff_tolerance'] for v in gains),
        relative_gain=value / baseline if baseline > 0 else None,
        baseline_loss=baseline, candidate_loss=candidate_metrics['objective'],
        used_as_additional_accuracy_reward=False, evidence='Same required folds, targets, loss components and normalization')
