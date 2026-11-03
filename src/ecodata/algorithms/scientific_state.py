"""Conservative full-domain state identity, separate from execution provenance."""
from ecodata.algorithms.dag import Artifact


TRANSPORT = {'source_file','source_sha256','processing_history','processing_stage',
             'time_resampling_action'}


def signature(dataset):
    value=dataset.copy(deep=False)
    value.attrs={k:v for k,v in dataset.attrs.items() if k not in TRANSPORT}
    # Preserve unknown attrs, all variables, physical units, interval supports,
    # source IDs and calibration flags. Never merge solely on validation values.
    return Artifact('scientific_state_v1',value).fingerprint()
