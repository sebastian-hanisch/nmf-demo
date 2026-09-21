"""Szenario (bit-identisch zum Vorgänger, neu: Synchronität), eingefrorene Vergleichsverfahren, Zuordnung und Kennzahlen (Handinstanzen), Urteil, Ansichten."""

import numpy as np
import pytest

import nm_algorithm as alg
import nm_constants as C
import nm_evaluation as ev
import nm_scenario as sc


# --- Szenario ------------------------------------------------------------------------------------------------------------------------


def test_default_dataset_is_bit_identical_to_the_predecessor_demos():
    ds = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7)
    assert float(ds.X.sum()) == 20.409393146494438 and sum(len(t) for t in ds.spike_times) == 227 and ds.X.shape == (4, 20000)
    assert (ds.A > 0).all()


def test_synchrony_zero_changes_nothing_and_nonzero_couples_the_neurons():
    a = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7)
    b = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7, 0.0)
    assert np.array_equal(a.X, b.X)
    s = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7, 0.9)
    lead = set(s.spike_times[0].tolist())
    for i in (1, 2, 3):
        shared = len(lead & set(s.spike_times[i].tolist()))
        assert shared >= 0.7 * len(s.spike_times[i])                                    # der größte Teil der Spikes fällt mit denen von Neuron 1 zusammen
    assert all(len(t) > 20 for t in s.spike_times)
    assert len(set(a.spike_times[0].tolist()) & set(a.spike_times[1].tolist())) < 3
    assert np.array_equal(sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7, 0.9).X, s.X)         # deterministisch


def test_similarity_two_widens_the_width_differences():
    assert sc.neuron_sigma(0, 2.0) != sc.neuron_sigma(0, 1.0)
    assert abs(sc.neuron_sigma(1, 2.0) - sc.neuron_sigma(0, 2.0)) > abs(sc.neuron_sigma(1, 1.0) - sc.neuron_sigma(0, 1.0))


# --- Eingefrorene Vergleichsverfahren (wortgleich aus den Vorgänger-Demos) ----------------------------------------------------------------


def test_comparators_are_frozen_at_the_predecessor_values():
    ds = sc.make_dataset(4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 7)
    a = ev.analyse(ds, ev.Settings())
    assert a.scores["ica"]["f1"] == 1.0 and a.scores["sobi"]["f1"] == 1.0 and a.scores["sca"]["f1"] == 1.0
    assert a.scores["ica"]["corr"] == pytest.approx(0.9807535100862836, abs=1e-9)
    assert a.scores["sobi"]["corr"] == pytest.approx(0.977758457497623, abs=1e-9)
    assert a.scores["sca"]["corr"] == pytest.approx(0.9951351889060601, abs=1e-9)
    assert a.scores["pca"]["f1"] == pytest.approx(0.6075434539940039, abs=1e-9)
    assert a.scores["electrode"]["f1"] == pytest.approx(0.6537682070375624, abs=1e-9)


# --- Zuordnung und Kennzahlen -------------------------------------------------------------------------------------------------------


def test_mixing_cos_hand_instances():
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert ev.mixing_cos(A, A * np.array([3.0, 0.5])) == pytest.approx(1.0)                # Skala egal
    assert ev.mixing_cos(A, A[:, ::-1]) == pytest.approx(1.0)                              # Reihenfolge egal
    assert ev.mixing_cos(A, A[:, :1]) == pytest.approx(0.5)                                # ein Neuron ohne Partner zählt 0
    B = np.eye(2)
    assert ev.mixing_cos(B, np.array([[1.0, 1.0], [1e-9, 1e-9]])) == pytest.approx(0.5 * (1 + 1e-9 / np.sqrt(2)) + 0.0, abs=0.36)


def test_column_similarity_hand_instance():
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert ev.column_similarity(A) == pytest.approx(np.sqrt(0.5))
    assert ev.column_similarity(np.eye(3)) == 0.0
    assert ev.column_similarity(np.array([[1.0], [2.0]])) == 0.0


def test_assign_and_matched_recover_a_permutation_with_sign_and_scale():
    rng = np.random.default_rng(0)
    S = rng.standard_normal((3, 500))
    est = np.array([-2.0 * S[2], 0.5 * S[0], S[1]]) + 0.01 * rng.standard_normal((3, 500))
    idx, corr, aligned = ev.matched(S, est)
    assert list(idx) == [1, 2, 0] and (corr > 0.99).all()
    assert np.allclose(aligned, S, atol=0.1)


def test_spike_f1_and_detection_on_hand_instances():
    assert ev.spike_f1(np.array([100, 300]), np.array([102, 298])) == 1.0
    assert ev.spike_f1(np.array([100]), np.array([100, 300])) == pytest.approx(2 / 3)
    assert ev.spike_f1(np.array([], dtype=int), np.array([100])) == 0.0
    x = 0.1 * np.random.default_rng(0).standard_normal(1000)
    times = np.arange(50, 1000, 80)                                      # zwölf Spitzen: die typische Tiefe (Median der zehn tiefsten) hebt die Schwelle über das Rauschen
    x[times] = -3.0
    assert list(ev.detect_spikes(x)) == list(times)


def test_single_electrode_has_no_mixing_angle_and_more_components_than_electrodes_is_allowed():
    a = ev.analyse_for((2, 1, 1.0, 2.0, 0.0, 0.05, 20000, 0.0, 7), ev.Settings(), comparators=False)
    assert a.single and np.isnan(a.scores["nmf"]["angle"]) and np.isnan(a.error_true) and a.V.shape[0] == C.SPECTROGRAM_WINDOW // 2 + 1 and a.sources.shape == (2, 20000)
    b = ev.analyse_for((5, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7), ev.Settings(), comparators=False)
    assert b.k == 5 and b.nmf.W.shape == (4, 5)


def test_analysis_is_deterministic_and_k_unknown_uses_the_elbow():
    p = (3, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7)
    a, b = ev.analyse_for(p, ev.Settings(), comparators=False), ev.analyse_for(p, ev.Settings(), comparators=False)
    assert a.scores["nmf"]["f1"] == b.scores["nmf"]["f1"] and a.nmf.error == b.nmf.error
    u = ev.analyse_for(p, ev.Settings(k_mode="unknown"), comparators=False)
    assert u.k == 3 and len(u.curve) >= 3


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------


def _code(params, **settings):
    return ev.verdict(ev.analyse_for(params, ev.Settings(**settings)))[1]


def test_verdict_codes_for_the_presets_and_the_edge_cases():
    assert _code((4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7)) == "others_win"
    assert _code((4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.9, 7)) == "nmf_wins"
    assert _code((2, 1, 1.0, 2.0, 0.0, 0.05, 20000, 0.0, 7)) == "single_works"
    assert _code((2, 1, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7)) == "single_weak"
    assert _code((5, 6, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7), k_mode="unknown") == "wrong_k"
    assert _code((4, 2, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7)) == "too_few_electrodes"
    assert _code((4, 4, 1.0, 1.0, 0.0, 1.0, 20000, 0.0, 7), threshold=3.0) in ("noise", "others_win")


def test_verdict_kind_matches_code():
    for p in ((4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.9, 7), (2, 1, 1.0, 2.0, 0.0, 0.05, 20000, 0.0, 7)):
        kind, code, data = ev.verdict(ev.analyse_for(p, ev.Settings()))
        assert kind == "success" and code in ("nmf_wins", "single_works") and data["f1"] >= 0.0 and data["m"] == p[0]


# --- Ansichten -----------------------------------------------------------------------------------------------------------------------


def test_component_view_and_progress_shapes():
    a = ev.analyse_for((4, 4, 1.0, 1.0, 0.0, 0.05, 20000, 0.0, 7), ev.Settings(), comparators=False)
    idx, cm, aligned, detected = ev.component_view(a)
    assert len(idx) == 4 and sorted(idx) == [0, 1, 2, 3] and cm.shape == (4, 4) and aligned.shape == (4, 20000) and len(detected) == 4
    its, corr, cos, f1 = ev.progress(a)
    assert its[0] == 0 and its == sorted(its) and len(corr) == len(cos) == len(f1) == len(its) and its[-1] == a.nmf.n_iter
    assert corr[-1] == pytest.approx(a.scores["nmf"]["corr"]) and f1[-1] == pytest.approx(a.scores["nmf"]["f1"]) and cos[-1] == pytest.approx(a.scores["nmf"]["angle"])
    one = ev.analyse_for((2, 1, 1.0, 2.0, 0.0, 0.05, 20000, 0.0, 7), ev.Settings(), comparators=False)
    assert all(np.isnan(c) for c in ev.progress(one)[2])


def test_sweep_rows_carry_mean_std_and_range_and_settings_sweeps_skip_the_comparators():
    rows = ev.sweep("threshold", values=(0.0, 2.0), comparators=False)
    assert [r["x"] for r in rows] == [0.0, 2.0] and all(("nmf" in r and "nmf_std" in r and "nmf_min" in r and "nmf_max" in r and "error_true" in r) for r in rows)
    assert all(r["nmf_min"] <= r["nmf"] <= r["nmf_max"] for r in rows) and "ica" not in rows[0]
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    full = ev.sweep("noise", values=(0.05,))
    assert "ica" in full[0] and "sca" in full[0]


def test_settings_and_data_defaults_agree_with_the_constants():
    s = ev.Settings()
    assert (s.mode, s.threshold, s.k_mode, s.init, s.sparsity, s.max_iter) == (C.DEFAULT_MODE, C.DEFAULT_THRESHOLD, C.DEFAULT_K_MODE, C.DEFAULT_INIT, C.DEFAULT_SPARSITY, C.DEFAULT_ITERATIONS)
    assert set(ev.DEFAULT_DATA) == set(ev.DATA_KEYS) and C.SWEEP_SEEDS == tuple(range(100000, 100005))
    assert alg.RECTIFY_MODES == ("negative", "absolute") and set(alg.INITS) == {"nndsvd", "random"}
