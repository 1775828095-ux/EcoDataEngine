import pytest
from ecodata.interface.camx_inventory_audit import audit_inventory


DIAG='''Area source species 1 NO mapped to model species 34 NO
AREA species: NR not modeled
Point source species 1 NO mapped to model species 34 NO
PT species: NR not modeled'''


def test_native_ignored_species_is_incomplete_not_a_successful_mapping():
    r=audit_inventory(['NO','NR'],DIAG)
    assert not r['complete'] and r['Area']['not_modeled']==['NR'] and r['Point']['mapped_count']==1
    with pytest.raises(ValueError,match='NR'):audit_inventory(['NO','NR'],DIAG,require_complete=True)


def test_unreported_missing_species_cannot_be_silently_ignored():
    with pytest.raises(ValueError,match='无明确'):audit_inventory(['NO','NR'],DIAG.replace('PT species: NR not modeled',''))


def test_only_complete_both_source_types_can_pass_complete_contract():
    diag=DIAG.replace('AREA species: NR not modeled','').replace('PT species: NR not modeled','')
    assert audit_inventory(['NO'],diag,require_complete=True)['complete']


def test_duplicate_source_species_rejected():
    with pytest.raises(ValueError,match='重复'):audit_inventory(['NO','NO'],DIAG)
