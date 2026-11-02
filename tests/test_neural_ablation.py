from dataclasses import replace

import numpy as np
import pytest
import torch
import xarray as xr

from ecodata.core.exceptions import SchemaError
from ecodata.fusion.observation_encoder import AttentionObservationEncoder
from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig, run_neural_reconstruction
from test_observation_guided_experiment import _sample


def test_main_attention_and_residual_can_be_removed_independently():
    torch.manual_seed(17)
    encoder = AttentionObservationEncoder(4, 2, residual_gate=.5, cross_attention=False)
    points = torch.tensor([[[0., 0., 1., 1.], [1., 1., 2., .5]]])
    valid = torch.tensor([[True, True]])
    queries = torch.tensor([[0., 0.], [.5, .5]])
    grid = torch.zeros(1, 2, 1, 2)
    residual = encoder(points, valid, queries, grid)
    assert residual.abs().sum() > 0
    # Removing background-query attention really removes grid dependence.
    torch.testing.assert_close(residual, encoder(points, valid, queries, grid + 100))
    encoder.residual_gate = 0
    assert torch.equal(encoder(points, valid, queries, grid), torch.zeros_like(residual))
    encoder.cross_attention = True
    assert encoder(points, valid, queries, grid).abs().sum() > 0
    assert torch.equal(encoder(points, ~valid, queries, grid), torch.zeros_like(residual))


@pytest.mark.parametrize('changes', [dict(cross_attention=False), dict(observation_residual_gate=0),
    dict(spectral_weight=0), dict(peak_weight=0), dict(reliability_weighting=False), dict(backbone='fno')])
def test_ablation_checkpoint_and_test_label_isolation(tmp_path, changes):
    bg, _, stations = _sample(1)
    stations = stations.copy()
    # One excluded station carries a deliberately absurd test value.
    extra = stations[stations.split == 'validation'].iloc[:1].copy()
    extra['station_id'] = 'unseen-test'
    extra['split'] = 'test'
    extra['value'] = 1e8
    import pandas as pd
    with_test = pd.concat([stations, extra], ignore_index=True)
    config = replace(NeuralReconstructionConfig(backend='observation_guided_uno',
        epochs=1, hidden_channels=4, n_layers=1, save_checkpoint_path=str(tmp_path/'best.pt')), **changes)
    trained = run_neural_reconstruction(bg, with_test, config=config)
    inferred = run_neural_reconstruction(bg, stations, config=replace(config, training_mode='inference',
        load_checkpoint_path=str(tmp_path/'best.pt'), save_checkpoint_path=None))
    xr.testing.assert_equal(trained.field, inferred.field)
    assert trained.report['test_station_records_excluded'] == 1
    assert not trained.report['has_independent_full_field_evaluation']
    if not config.reliability_weighting:
        assert not trained.report['observation_reliability']['enabled']
    if config.backbone == 'fno':
        assert trained.report['ablation_controls']['backbone'] == 'fno'
        assert trained.report['ablation_controls']['peak_loss_weight'] > 0
    if not config.cross_attention:
        with pytest.raises(SchemaError, match='不兼容'):
            run_neural_reconstruction(bg, stations, config=replace(config, cross_attention=True,
                training_mode='inference', load_checkpoint_path=str(tmp_path/'best.pt')))


def test_old_signature_checkpoint_keeps_default_semantics(tmp_path):
    bg, _, stations = _sample(1)
    path = tmp_path/'legacy.pt'
    config = NeuralReconstructionConfig(epochs=1, hidden_channels=4, n_layers=1,
                                        save_checkpoint_path=str(path))
    original = run_neural_reconstruction(bg, stations, config=config)
    checkpoint = torch.load(path, weights_only=False)
    for key in ('backbone', 'cross_attention', 'reliability_weighting', 'observation_residual_gate'):
        checkpoint['signature'].pop(key)
    for key in ('backbone', 'cross_attention', 'reliability_weighting'):
        checkpoint['config'].pop(key)
    torch.save(checkpoint, path)
    reused = run_neural_reconstruction(bg, stations, config=replace(config, training_mode='inference',
        load_checkpoint_path=str(path), save_checkpoint_path=None))
    np.testing.assert_array_equal(original.field.value, reused.field.value)
