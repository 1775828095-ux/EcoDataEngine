"""Historical pure semantic weighting helper; GOLEM search has been retired."""
def semantic_proposal_weights(decision, factory, before, candidates, *, soft=True, alpha=2., beta=4.):
    """Shared typed numerical-search guidance, NOT the legacy structural objective.

    Only gap/support evidence changes proposal probabilities; numerical fitness
    still comes exclusively from the real executor's independent evaluator.
    """
    import numpy as np
    current = decision.assess_graph(factory,before)
    assessments = [decision.assess_graph(factory,c) for c in candidates]
    use_soft=soft and "retained_observation_support" in decision.requirements.soft_objectives
    scores=[alpha*(current["gap"]-a["gap"])-
            (beta*a["support_deficit"] if use_soft and a["support_deficit"] is not None else 0.) for a in assessments]
    weights=np.exp(np.clip(scores,-20,20));probabilities=weights/weights.sum() if len(weights) else weights
    return weights.tolist(),dict(before=current,candidates=assessments,weights=weights.tolist(),
        probabilities=probabilities.tolist(),soft_enabled=use_soft,alpha=alpha,beta=beta,
        reason="shared requirements + output-path capabilities + observable support; no accuracy labels")
