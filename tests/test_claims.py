"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Datensätze belegt (Mittel; Toleranz ±0.015 = Rundung auf zwei Stellen plus etwas Luft).
Positive UND negative Aussagen: wo die NMF verliert, steht das hier ebenso als Test wie dort, wo sie gewinnt."""

from functools import lru_cache

import numpy as np
import pytest

import nm_algorithm as alg
import nm_constants as C
import nm_evaluation as ev

TOL = 0.015


@lru_cache(maxsize=None)
def _runs(items, settings, comparators):
    return tuple(ev.analyse(ev.make_dataset(seed=s, **dict(items)), settings, comparators=comparators) for s in C.SWEEP_SEEDS)


def runs(settings=ev.Settings(), comparators=True, **kw):
    return _runs(tuple(sorted(kw.items())), settings, comparators)


def f1(name="nmf", settings=ev.Settings(), comparators=True, **kw):
    return float(np.mean([a.scores[name]["f1"] for a in runs(settings, comparators, **kw)]))


def corr(name="nmf", settings=ev.Settings(), comparators=True, **kw):
    return float(np.mean([a.scores[name]["corr"] for a in runs(settings, comparators, **kw)]))


def base(a):
    return float(np.mean([r.baseline_f1 for r in a]))


def near(value, expected, tol=TOL):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall: NMF trennt schlechter als die Vorgänger, findet aber die besser passende Zerlegung ------------------------------------------


def test_default_nmf_is_clearly_worse_than_ica_sobi_and_sca_but_fits_better_than_the_truth():
    r = runs()
    near(f1(), 0.77)
    for name in ("ica", "sobi", "sca"):
        near(f1(name), 1.00)
    near(np.mean([a.nmf.error for a in r]), 0.013, 0.003)
    near(np.mean([a.error_true for a in r]), 0.049, 0.003)
    assert all(a.nmf.error < a.error_true for a in r)                              # besser passend, aber falsch
    near(np.mean([a.column_similarity for a in r]), 0.90, 0.03)
    near(np.mean([a.scores["nmf"]["angle"] for a in r]), 0.95, 0.02)


@pytest.mark.parametrize("m,expected,ica", [(2, 0.87, 1.00), (3, 0.84, 1.00), (4, 0.77, 1.00), (5, 0.59, 0.77)])
def test_neuron_count_sweep(m, expected, ica):
    near(f1(m=m), expected)
    near(f1("ica", m=m), ica)
    if m < 5:
        assert f1("sca", m=m) >= 0.98
    else:
        near(f1("sca", m=5), 0.99)


@pytest.mark.parametrize("n,nmf,ica,sobi,sca", [(2, 0.64, 0.36, 0.35, 0.97), (3, 0.72, 0.69, 0.69, 0.99), (4, 0.77, 1.00, 1.00, 1.00), (6, None, 1.0, 1.0, 1.0)])
def test_electrode_count_sweep(n, nmf, ica, sobi, sca):
    if nmf is not None:
        near(f1(n=n), nmf)
    near(f1("ica", n=n), ica)
    near(f1("sobi", n=n), sobi)
    near(f1("sca", n=n), sca, 0.03)


def test_one_electrode_at_four_neurons():
    a = runs(n=1)
    near(f1(n=1), 0.38)
    near(base(a), 0.40)
    near(f1("ica", n=1), 0.16)
    near(f1("sobi", n=1), 0.16)
    near(f1("sca", n=1), 0.05)


@pytest.mark.parametrize("rate,expected", [(0.25, 0.78), (0.5, 0.80), (1.0, 0.77), (2.0, 0.79)])
def test_rate_sweep(rate, expected):
    near(f1(comparators=False, rate_scale=rate), expected, 0.02)


def test_similarity_sweep_multi_electrode_and_sobi():
    near(f1(comparators=False, similarity=0.0), 0.77, 0.02)
    near(f1(comparators=False, similarity=2.0), 0.76, 0.02)
    near(f1("sobi", similarity=0.0), 0.58, 0.02)


def test_single_electrode_needs_very_different_widths():
    for sim, expected, thr in ((1.0, 0.50, 0.66), (2.0, 0.97, 0.66)):
        r = runs(m=2, n=1, similarity=sim)
        near(f1(m=2, n=1, similarity=sim), expected, 0.02)
        near(base(r), thr, 0.02)
    assert f1(m=2, n=1, similarity=1.0) < base(runs(m=2, n=1, similarity=1.0))          # negativ: unter der bloßen Schwelle
    assert f1(m=2, n=1, similarity=2.0) - base(runs(m=2, n=1, similarity=2.0)) > 0.25       # positiv
    for name, expected in (("ica", 0.33), ("sobi", 0.33), ("sca", 0.04)):
        near(f1(name, m=2, n=1, similarity=2.0), expected, 0.02)


@pytest.mark.parametrize("jitter,expected", [(0.0, 0.77), (0.1, 0.80), (0.2, 0.81), (0.3, 0.81)])
def test_jitter_sweep(jitter, expected):
    near(f1(comparators=False, jitter=jitter), expected, 0.02)
    if jitter == 0.3:
        for name in ("ica", "sobi", "sca"):
            assert f1(name, jitter=0.3) >= 0.98


@pytest.mark.parametrize("sync,nmf,ica,sobi,sca", [(0.0, 0.77, 1.00, 1.00, 1.00), (0.3, 0.87, 1.00, 1.00, 1.00), (0.6, 0.91, 0.94, 0.99, 0.92), (0.9, 0.98, 0.66, 0.80, 0.72)])
def test_synchrony_sweep(sync, nmf, ica, sobi, sca):
    near(f1(synchrony=sync), nmf, 0.02)
    near(f1("ica", synchrony=sync), ica, 0.02)
    near(f1("sobi", synchrony=sync), sobi, 0.02)
    near(f1("sca", synchrony=sync), sca, 0.02)


def test_synchrony_caveat_best_single_electrode_is_strong_too():
    near(f1("electrode", synchrony=0.9), 0.92, 0.02)
    near(corr("electrode", synchrony=0.9), 0.94, 0.02)
    near(corr(synchrony=0.9), 0.87, 0.02)
    for name, expected in (("ica", 0.56), ("sobi", 0.68), ("sca", 0.64)):
        near(corr(name, synchrony=0.9), expected, 0.03)


@pytest.mark.parametrize("noise,nmf,ica,sobi,sca", [(0.05, 0.77, 1.00, 1.00, 1.00), (0.2, 0.75, 0.99, 0.79, 1.00), (0.5, 0.71, 0.83, 0.53, 0.94), (1.0, 0.71, 0.43, 0.38, 0.86)])
def test_noise_sweep(noise, nmf, ica, sobi, sca):
    s = ev.Settings(threshold=2.0)
    near(f1(settings=s, noise=noise), nmf, 0.02)
    near(f1("ica", settings=s, noise=noise), ica, 0.02)
    near(f1("sobi", settings=s, noise=noise), sobi, 0.02)
    near(f1("sca", settings=s, noise=noise), sca, 0.02)


def test_rectification_mode_costs_correlation_not_f1():
    near(f1(comparators=False), 0.77)
    near(f1(settings=ev.Settings(mode="absolute"), comparators=False), 0.78, 0.02)
    near(corr(comparators=False), 0.79, 0.02)
    near(corr(settings=ev.Settings(mode="absolute"), comparators=False), 0.55, 0.03)


def test_noise_threshold_barely_matters_at_low_noise_and_decides_at_high_noise():
    near(f1(settings=ev.Settings(threshold=0.0), comparators=False), 0.76, 0.02)
    for thr, expected in ((0.0, 0.52), (1.0, 0.63), (2.0, 0.71), (3.0, 0.74)):
        near(f1(settings=ev.Settings(threshold=thr), comparators=False, noise=1.0), expected, 0.02)


def test_sparsity_penalty_helps_a_little_and_raises_the_error():
    r0 = runs(settings=ev.Settings(sparsity=0.0), comparators=False)
    for lam, expected, cos in ((0.5, 0.81, 0.98), (1.0, 0.80, 0.98), (3.0, 0.79, 0.98)):
        r = runs(settings=ev.Settings(sparsity=lam), comparators=False)
        near(f1(settings=ev.Settings(sparsity=lam), comparators=False), expected, 0.02)
        near(np.mean([a.scores["nmf"]["angle"] for a in r]), cos, 0.02)
    r3 = runs(settings=ev.Settings(sparsity=3.0), comparators=False)
    near(np.mean([a.nmf.error for a in r3]), 0.086, 0.01)
    assert np.mean([a.nmf.error for a in r3]) > np.mean([a.error_true for a in r3])            # dann nicht mehr besser passend als die Wahrheit
    assert np.mean([a.nmf.error for a in r0]) < np.mean([a.error_true for a in r0])


def test_more_iterations_fit_better_but_separate_worse():
    errs, f1s = {}, {}
    for it in (20, 100, 300, 1000):
        s = ev.Settings(max_iter=it)
        errs[it] = float(np.mean([a.nmf.error for a in runs(settings=s, comparators=False)]))
        f1s[it] = f1(settings=s, comparators=False)
    near(errs[20], 0.055, 0.005)
    near(errs[100], 0.017, 0.003)
    near(errs[1000], 0.010, 0.003)
    assert errs[20] > errs[100] > errs[300] > errs[1000]
    for it, expected in ((20, 0.75), (100, 0.80), (300, 0.77), (1000, 0.74)):
        near(f1s[it], expected, 0.02)
    assert f1s[1000] < f1s[100]                                                                   # negativ: mehr Anpassung, schlechtere Trennung


def test_initialisation_random_vs_nndsvd_and_the_spread_of_starts():
    near(f1(comparators=False), 0.77)
    near(f1(settings=ev.Settings(init="nndsvd"), comparators=False), 0.73, 0.02)
    t = ev.init_table()
    near(t["mean"], 0.76, 0.03)
    near(t["std"], 0.03, 0.015)
    near(t["lowest"], 0.77, 0.03)
    near(t["nndsvd"], 0.73, 0.03)
    assert t["err_nndsvd"] > t["err_mean"]
    assert t["worst"] < t["best"]


def test_rank_choice_hits_two_to_four_neurons_and_misses_five():
    rows = {r["m"]: r for r in ev.rank_table()}
    for m in (2, 3, 4):
        assert rows[m]["chosen"] == [m] * 5, rows[m]["chosen"]
    assert rows[5]["chosen"] == [4] * 5, rows[5]["chosen"]
    assert rows[5]["correct"] == 0.0


def test_five_neurons_unknown_k_preset_numbers():
    s = ev.Settings(k_mode="unknown")
    near(f1(settings=s, m=5, n=6), 0.51, 0.03)
    near(f1(settings=ev.Settings(), m=5, n=6), 0.61, 0.03)
    near(f1("ica", settings=s, m=5, n=6), 0.86, 0.03)
    near(f1("sca", settings=s, m=5, n=6), 0.99, 0.02)


def test_strong_noise_preset_numbers():
    s = ev.Settings(threshold=3.0)
    near(f1(settings=s, comparators=True, noise=1.0), 0.74, 0.02)
    near(f1(settings=ev.Settings(threshold=0.0), comparators=False, noise=1.0), 0.52, 0.02)
    near(f1("ica", settings=s, noise=1.0), 0.43, 0.03)
    near(f1("sobi", settings=s, noise=1.0), 0.38, 0.03)
    near(f1("sca", settings=s, noise=1.0), 0.86, 0.03)


def test_scene_table_matches_the_chart():
    rows = {r["scene"]: r for r in ev.scene_table()}
    assert len(rows) == 8
    std = rows["Standardfall (4 Neuronen, 4 Elektroden)"]
    near(std["nmf"], 0.77)
    near(std["sca"], 1.00)
    syn = rows["Synchrones Feuern (90 % gemeinsame Spikes)"]
    assert syn["nmf"] > max(syn["ica"], syn["sobi"], syn["sca"]) + 0.15
    noisy = rows["Starkes Rauschen (1.0)"]
    assert noisy["ica"] < noisy["nmf"] < noisy["sca"]
    assert noisy["sobi"] < noisy["nmf"]
    two = rows["Zwei Elektroden, vier Neuronen"]
    near(two["nmf"], 0.64)
    near(two["sca"], 0.97)
    five = rows["Fünf Neuronen auf vier Elektroden"]
    near(five["nmf"], 0.59)
    assert five["sca"] > five["ica"] > five["nmf"]
    eight = rows["Zwei Neuronen, acht Elektroden"]
    near(eight["nmf"], 0.87)
    weak = rows["Eine Elektrode: Breiten wie gewohnt (2 Neuronen)"]
    assert weak["nmf"] < weak["baseline"]
    strong = rows["Eine Elektrode: sehr verschiedene Breiten (2 Neuronen, Ähnlichkeit 2)"]
    near(strong["nmf"], 0.97, 0.02)
    assert strong["nmf"] > strong["baseline"] + 0.25


def test_true_factorisation_error_is_a_valid_reference():
    ds = ev.make_dataset(noise=0.0)
    V = alg.rectify(ds.X, "negative", 0.0)
    assert ev.true_factorisation_error(V, ds.A) < 0.2
