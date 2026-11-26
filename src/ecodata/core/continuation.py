"""Step-scoped continuation receipts; advice cannot override scientific gates.

Callers classify requirements using their existing contracts.  This module never
infers units, coordinates, observations, or model configuration from an error.
"""


def decision(stage, target, reason, *, required=True, recovery=None, evidence=None, advice=None):
    action = 'BLOCK_STEP' if required else 'CONTINUE_WITH_WARNING'
    # A recovery must already have been checked by the producing service.
    if recovery and recovery.get('verified') is True:
        if recovery.get('action') not in {'AUTO_COMPLETED', 'FALLBACK', 'SKIP_STEP'}:
            raise ValueError('未登记的补救操作')
        if required and recovery['action'] == 'SKIP_STEP':
            raise ValueError('必要条件不能通过跳过检查变成已满足')
        action = recovery['action']
    applied = False
    if not required and advice and advice.get('action') == 'SKIP_STEP':
        action, applied = 'SKIP_STEP', True
    return dict(stage=stage, target=str(target), reason=str(reason), required=required,
                action=action, evidence=evidence or [], recovery=recovery,
                advice=advice, advice_applied=applied,
                blocks_only=[str(target)] if action == 'BLOCK_STEP' else [],
                scientific_data_modified=False)


def check_decision(row, rows):
    """DS suggestions are advisory; actual contract failures stay local blocks."""
    deferred = (row.get('action') == 'quality_screen'
        and row.get('next_step') == 'runtime_quality_gate'
        and any(other.get('candidate') == row.get('candidate')
                and other.get('next_step') == 'original_workbench' for other in rows))
    admissible = row.get('status') in {'通过', '无需新增处理'} or row.get('next_step') == 'original_workbench' or deferred
    skipped = row.get('disposition') in {'SKIPPED_NOT_PROCESSING_VARIABLE', 'SKIPPED_UNSUPPORTED_PROPOSAL'}
    # No new operation is justified by an unresolved goal. Other verified
    # alignment/conversion operations remain usable, without inventing fusion.
    optional = skipped or row.get('action') in {'pending_goal', 'insufficient', 'reject'}
    return decision('proposal', str(row.get('candidate', 'task')) + ':' + str(row.get('action', 'check')),
                    row.get('reason') or row.get('status', 'UNKNOWN'),
                    required=not (admissible or optional),
                    recovery=dict(action='SKIP_STEP', verified=True) if optional else None,
                    evidence=[row.get('block_code')] if row.get('block_code') else [],
                    advice=dict(action='SKIP_STEP', source='DS proposal')
                        if row.get('action') in {'pending_goal','insufficient','reject'} else None)


def receipt(events, completed=()):
    events = list(events)
    completed = list(completed)
    blocked = [r for r in events if r['action'] == 'BLOCK_STEP']
    return dict(schema='ecodata.continuation.v1',
                status='PARTIAL' if blocked and completed else 'BLOCKED' if blocked else 'COMPLETED_WITH_NOTES' if completed else 'NOT_RUN',
                completed_steps=completed, decisions=events,
                blocked_steps=[r['target'] for r in blocked],
                independent_steps_continue=True, original_inputs_modified=False,
                qualification='步骤处置记录，不代表精度提升、完整工程可运行或模型已执行')


def final_events(report):
    events=[];completed=[]
    for key in ('format', 'project', 'native'):
        section=report.get(key, {});status=section.get('status', 'UNKNOWN')
        if status == 'PASS':
            completed.append(key)
        elif status not in {'NOT_REQUESTED'}:
            events.append(decision('final_check', key,
                section.get('reason') or '；'.join(map(str, section.get('errors', []))) or status,
                required=key != 'native', evidence=[dict(status=status)]))
    return events,completed
