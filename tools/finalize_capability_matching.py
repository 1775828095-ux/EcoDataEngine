"""Freeze current source/evidence manifests, no search/training/data modification."""
from pathlib import Path
import sys,json,hashlib,shutil,xml.etree.ElementTree as ET,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from ecodata.algorithms.numerical_search import save
OUT=ROOT/'output/capability_matching_20261021'


def run():
    rows=[]
    def row(item,status,source,evidence,limit=''):
        rows.append(dict(item=item,status=status,source=source,evidence=evidence,limit=limit))
    row('variable_class / unit / spatial_support / temporal_semantics','已实现且有行为证据','semantic_requirements.py / semantic_rules.json','test_capability_scoring.py; layers/*/decision.json','Scientific meanings need normal metadata, not inferred from values')
    row('grid relation, local source identity and affected port','已实现且有行为证据','router.py:bind_problem_ports','verification.json:affected_problems/background_preserved','Derived resolutions of one product are not independent data sources')
    row('finite/missing/negative counts and per-series visible support','已实现且有行为证据','SemanticFeatureExtractor.extract/extract_series','tests_affected.xml; tests_final.xml','No labels or private evaluation masks; hourly splitter remains hourly-only')
    row('12 raster shape/scale statistics','已实现且有行为证据','SemanticFeatureExtractor.raster_features / raster_upload.input_features','tests_score_final.xml; tests_affected.xml; model_report.json','Only finite >=3x3 2D inputs; old feature formula reused, no class label')
    row('required capabilities and explicit mandatory requests','已实现且有行为证据','RequirementMapper / SemanticDecision.require_graph','test_required_vs_optional_and_unknown_score_not_zero; decision boundary tests','Actual output-path and state proofs; no global union')
    row('optional consistency and paired train bias diagnosis','已实现且有行为证据','semantic_rules.json / capability_scoring.py','test_legal_training_bias_vs_insufficient_reference; decision.json','Training bias diagnosis is not measured application gain')
    row('implementation-conditional empirical benefit','已实现且有行为证据','CapabilityScoringService / numerical_search.problem_directed','pairs.json; runs/*_v4/summary.json','Development-only RF, 3 regions; deployment heldout 0/12 in support')
    row('trusted/generalizable optional gain prediction','当前证据不足，不能可靠推断','model_diagnostics.json','deployment_holdout_in_support=0; heldout_regions=1','C acceptance remains insufficient; no altered support thresholds')
    row('other optional capabilities empirical scores','确实缺失','optional_capability_matches','Scores remain null without implementation-specific model/evidence','No empty models, default high benefits, or false zero benefits')
    row('declared facts vs auto identification same consumer','已实现且有行为证据','apply_declarations / submit_search / TemporalEvaluator.decision','verification.json.comparisons; tests_score_final.xml','Conflict rejects; unknown remains unknown; no feature winner fields')
    row('C-local graph edits and scoring probabilities','已实现且有行为证据','problem_candidates / empirical_proposal_weights','verification.json.score_probability_max_change; proposal actual chosen_index','Old A/B/C/C-gap/R/C-directed policy preserved; no superiority experiment')
    row('existing KNN raster error router','已实现但未核实','raster_upload.route','Earlier ch36 evidence and current formula regression only','Not retrained or relabeled as capability gain model')
    row('curvilinear/terrain/neural/model-preprocessing generality','已实现但未核实','Existing registry/adapter contracts, ch41 matrix','No new end-to-end run this round','Weights, input contract and native verification conditions retained')
    save(OUT/'feature_capability_audit.json',rows)
    suites=['tests_affected.xml','tests_final.xml','tests_mean_fixed.xml','tests_score_final.xml'];tests=set();batches=[]
    for name in suites:
        tree=ET.parse(OUT/name);suite=tree.find('testsuite')
        assert suite.attrib['failures']==suite.attrib['errors']==suite.attrib['skipped']=='0'
        tests.update((t.attrib.get('classname'),t.attrib['name']) for t in tree.iter('testcase'));batches.append(dict(file=name,**suite.attrib))
    files=[p for directory in ['src/ecodata/algorithms','src/ecodata/interface'] for p in (ROOT/directory).glob('*.py') if p.name in {
        'capability_scoring.py','semantic_requirements.py','router.py','golem_search.py','numerical_search.py','platform_search.py','temporal_search.py','plugins.py','raster_upload.py','search_page.py'}]
    files += [ROOT/'src/ecodata/algorithms/semantic_rules.json',ROOT/'tests/test_capability_scoring.py',ROOT/'tests/test_post_conversion_validation.py']
    files += list(ROOT.glob('tools/*capability_matching.py'))+[ROOT/'tools/accept_capability_layers.py']
    hashes={}
    for file in files:
        target=OUT/'source_executed'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
        hashes[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
    import sklearn,numpy
    save(OUT/'release.json',dict(version='two_class_development_v1/platform_search_v5',source_sha256=hashes,python=platform.python_version(),sklearn=sklearn.__version__,numpy=numpy.__version__,
        tests=dict(unique_passed=len(tests),batches=batches,failures_preserved='tests_mean.xml: 3 stale UI tests, explicitly navigated correct page and retested; interval_mean first run failed and preserved'),
        acceptance=dict(A='PASS',B='PASS',C='INSUFFICIENT_EVIDENCE',D='PASS (bounded supported cases)',E='PASS (bounded supported cases)',F='PASS (handler/toolkit/search/product ZIP)',G='PASS (recorded scope, no total-cost advantage claim)'),
        scope='No new algorithms or official study. No trusted benefit claim, no browser acceptance this round.',
        snapshots='Core files copied before editing; raster_upload.py pre-move body reconstructed from initial read, late captured version also retained; not a byte-identical raw snapshot claim'))


if __name__=='__main__':run()
