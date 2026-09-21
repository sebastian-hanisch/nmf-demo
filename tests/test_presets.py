"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import nm_constants as C
from nm_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["m"], p["n"], p["rate_scale"], p["similarity"], p["jitter"], p["noise"], p["n_samples"], p["synchrony"], p["seed"])
    a = analyse_for(params, Settings(p["mode"], p["threshold"], p["k_mode"], p["init"], p["sparsity"], p["iterations"]))
    return {"verdict": verdict(a)[1], "f1": a.scores["nmf"]["f1"], "k": a.k}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_most_presets_show_a_negative_result():
    negative = [n for n, b in C.PRESET_EXPECTED_BANDS.items() if set(b["verdict"]) & {"others_win", "single_weak", "wrong_k"}]
    assert len(negative) == 4


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_NEURONS_MIN <= p["m"] <= C.N_NEURONS_MAX and C.N_ELECTRODES_MIN <= p["n"] <= C.N_ELECTRODES_MAX
        assert C.RATE_SCALE_MIN <= p["rate_scale"] <= C.RATE_SCALE_MAX and abs(p["rate_scale"] * 4 - round(p["rate_scale"] * 4)) < 1e-9
        assert C.SIMILARITY_MIN <= p["similarity"] <= C.SIMILARITY_MAX and C.JITTER_MIN <= p["jitter"] <= C.JITTER_MAX and C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX
        assert C.SYNCHRONY_MIN <= p["synchrony"] <= C.SYNCHRONY_MAX and abs(p["synchrony"] * 20 - round(p["synchrony"] * 20)) < 1e-9
        assert C.N_SAMPLES_MIN <= p["n_samples"] <= C.N_SAMPLES_MAX and p["n_samples"] % 1000 == 0
        assert C.THRESHOLD_MIN <= p["threshold"] <= C.THRESHOLD_MAX and abs(p["threshold"] * 2 - round(p["threshold"] * 2)) < 1e-9
        assert C.SPARSITY_MIN <= p["sparsity"] <= C.SPARSITY_MAX and C.ITERATIONS_MIN <= p["iterations"] <= C.ITERATIONS_MAX and p["iterations"] % 10 == 0
        assert p["k_mode"] in C.K_MODES and p["mode"] in ("negative", "absolute") and p["init"] in ("nndsvd", "random")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if key == "verdict":
            assert value in expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
