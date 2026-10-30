import numpy as np
import pytest

from tools.evaluate_wetland_wad2m_stable import cellwise_metrics, stable_metrics


def test_stable_mask_excludes_cell_missing_in_any_month():
    model = np.array([[[0.1, 0.2]], [[0.3, 0.4]]])
    observed = np.ma.array(
        [[[0.2, 0.2]], [[0.1, 0.4]]],
        mask=[[[0, 0]], [[0, 1]]],
    )
    result = stable_metrics(model, observed, np.array([[2.0, 10.0]]))
    assert result["n_stable_cells"] == 1
    assert result["n_pairs"] == 2
    assert result["weighted_bias"] == pytest.approx(0.05)
    assert result["weighted_rmse"] == pytest.approx(np.sqrt(0.025))
    assert result["simulated_monthly_mean"] == pytest.approx([0.1, 0.3])
    assert result["n_reference_dynamic_cells"] == 1
    assert result["n_temporal_cells"] == 1
    assert result["weighted_mean_cell_temporal_correlation"] == pytest.approx(-1)
    assert result["weighted_temporal_correlation_on_reference_dynamic"] == pytest.approx(-1)
    assert result["weighted_positive_temporal_correlation_fraction"] == 0


def test_temporal_correlation_counts_constant_model_on_fixed_reference_mask():
    model = np.array([[[0.1, 0.2]], [[0.3, 0.2]]])
    observed = np.array([[[0.1, 0.2]], [[0.3, 0.3]]])
    result = stable_metrics(model, observed, np.array([[1.0, 3.0]]))
    assert result["n_reference_dynamic_cells"] == 2
    assert result["n_temporal_cells"] == 1
    assert result["weighted_mean_cell_temporal_correlation"] == pytest.approx(1)
    assert result["weighted_temporal_correlation_on_reference_dynamic"] == pytest.approx(0.25)


def test_cellwise_map_preserves_fixed_mask_and_constant_model():
    model = np.array([[[0.1, 0.2, 0.1]], [[0.3, 0.2, 0.2]]])
    observed = np.ma.array(
        [[[0.1, 0.2, 0.1]], [[0.3, 0.3, 0.2]]],
        mask=[[[0, 0, 0]], [[0, 0, 1]]],
    )
    result = cellwise_metrics(model, observed, np.ones((1, 3)))
    assert result["stable_cell"].tolist() == [[True, True, False]]
    assert result["reference_dynamic_cell"].tolist() == [[True, True, False]]
    assert result["rmse"][0, 0] == pytest.approx(0)
    assert result["rmse"][0, 1] == pytest.approx(np.sqrt(0.005))
    assert result["temporal_correlation"][0, 0] == pytest.approx(1)
    assert np.isnan(result["temporal_correlation"][0, 1])
    assert np.isnan(result["rmse"][0, 2])


def test_stable_mask_rejects_no_common_cell():
    with pytest.raises(ValueError, match="No cells"):
        stable_metrics(np.zeros((2, 1, 1)),
                       np.ma.masked_all((2, 1, 1)), np.ones((1, 1)))
