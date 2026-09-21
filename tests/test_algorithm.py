"""NMF-Kern: Kreuzprüfung gegen scikit-learn, exakte Rückgewinnung auf Handinstanzen, Struktur von NNDSVD, Gleichrichtung, Spektrogramm, Knick der Fehlerkurve."""

import numpy as np
import pytest
from sklearn.decomposition import NMF as SkNMF

import nm_algorithm as alg
import nm_constants as C


def _instance(F=6, k=3, T=80, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, (F, k))
    H = (rng.uniform(0, 1, (k, T)) > 0.6) * rng.uniform(0.5, 1.5, (k, T))
    return W, H, np.maximum(W @ H + noise * rng.standard_normal((F, T)), 0)


def test_error_is_monotone_non_increasing_and_factors_stay_non_negative():
    _, _, V = _instance(noise=0.05)
    r = alg.nmf_once(V, 3, "random", seed=1, max_iter=200)
    e = np.array(r.errors)
    assert (np.diff(e[:-1]) <= 1e-9).all()                                       # (der letzte Wert ist exakt nachgerechnet, davor die Spurformel)
    assert (r.W >= 0).all() and (r.H >= 0).all()
    assert np.allclose(r.W.sum(axis=0), 1.0)                                      # Spalten von W auf Summe 1, Skala in H
    assert r.error == pytest.approx(alg.relative_error(V, r.W, r.H))


def test_exact_recovery_of_a_noise_free_separable_instance():
    W = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.1], [0.0, 0.0, 0.7], [0.5, 0.5, 0.0]])
    rng = np.random.default_rng(3)
    H = (rng.uniform(0, 1, (3, 200)) > 0.7) * rng.uniform(0.5, 1.5, (3, 200))
    V = W @ H
    best, _ = alg.nmf(V, 3, "random", 0, 5, 2000, 1e-9)
    assert best.error < 0.02
    A = W / W.sum(axis=0)
    cos = np.abs((A / np.linalg.norm(A, axis=0)).T @ (best.W / np.linalg.norm(best.W, axis=0)))
    assert (cos.max(axis=1) > 0.99).all()                                         # jede wahre Spalte wird gefunden


def test_rank_one_instance_is_solved_by_one_component():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 1.0])
    V = np.outer(a, b)
    r = alg.nmf_once(V, 1, "random", seed=0, max_iter=300)
    assert r.error < 1e-6 and np.allclose(r.W[:, 0], a / a.sum())


def test_close_to_scikit_learn_in_error_on_a_hand_instance():
    _, _, V = _instance(noise=0.03, seed=5)
    sk = SkNMF(n_components=3, init="nndsvda", solver="mu", beta_loss="frobenius", max_iter=1000, tol=1e-9, random_state=0)
    Wk = sk.fit_transform(V)
    err_sk = np.linalg.norm(V - Wk @ sk.components_) / np.linalg.norm(V)
    ours, _ = alg.nmf(V, 3, "random", 0, 5, 1000, 1e-9)
    assert abs(ours.error - err_sk) < 0.02


def test_nndsvd_structure_and_determinism():
    _, _, V = _instance(noise=0.03)
    W, H = alg.nndsvd(V, 3)
    assert (W >= 0).all() and (H >= 0).all() and W.shape == (6, 3) and H.shape == (3, 80)
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    assert np.allclose(W[:, 0] / np.linalg.norm(W[:, 0]), np.abs(U[:, 0]), atol=1e-9)         # erste Komponente = führendes Singulärpaar
    W2, H2 = alg.nndsvd(V, 3)
    assert np.array_equal(W, W2) and np.array_equal(H, H2)
    Wz, _ = alg.nndsvd(V, 3, fill_mean=False)
    assert (Wz == 0).any() and not (W == 0).any()                                         # ohne Auffüllen Nullen, mit Auffüllen keine
    a, b = alg.nmf_once(V, 3, "nndsvd"), alg.nmf_once(V, 3, "nndsvd")
    assert a.error == b.error


def test_random_start_is_seeded_and_positive():
    _, _, V = _instance()
    W1, H1 = alg.random_start(V, 3, 5)
    W2, H2 = alg.random_start(V, 3, 5)
    W3, _ = alg.random_start(V, 3, 6)
    assert np.array_equal(W1, W2) and not np.array_equal(W1, W3) and (W1 > 0).all() and (H1 > 0).all()


def test_stopping_criterion_and_snapshots():
    _, _, V = _instance()
    loose = alg.nmf_once(V, 3, "random", seed=0, max_iter=1000, tol=1e-2)
    tight = alg.nmf_once(V, 3, "random", seed=0, max_iter=1000, tol=1e-9)
    assert loose.converged and loose.n_iter < tight.n_iter
    short = alg.nmf_once(V, 3, "random", seed=0, max_iter=20, tol=0.0)
    assert not short.converged and short.n_iter == 20 and len(short.errors) == 21
    assert set(short.snapshots) >= {0, 1, 5, 20} and 100 not in short.snapshots
    assert alg.relative_error(V, *short.snapshots[0]) == pytest.approx(short.errors[0])


def test_sparsity_penalty_yields_sparser_activities_and_larger_error():
    _, _, V = _instance(noise=0.02, seed=2)
    r0 = alg.nmf_once(V, 3, "random", seed=0, max_iter=300)
    r1 = alg.nmf_once(V, 3, "random", seed=0, max_iter=300, sparsity=3.0)
    assert r1.H.sum() < r0.H.sum() and r1.error > r0.error


def test_restarts_return_the_lowest_error_run():
    _, _, V = _instance(noise=0.05, seed=4)
    best, errors = alg.nmf(V, 3, "random", 10, 4, 100)
    assert len(errors) == 4 and best.error == min(errors)
    nn, errs = alg.nmf(V, 3, "nndsvd", 0, 4, 100)
    assert len(errs) == 1


def test_rectify_hand_instance_and_noise_threshold():
    X = np.array([[0.0, -1.0, 0.5, 0.0, 0.0], [0.0, 2.0, -3.0, 0.0, 0.0]])
    neg = alg.rectify(X, "negative", 0.0)
    assert np.array_equal(neg, np.array([[0, 1.0, 0, 0, 0], [0, 0, 3.0, 0, 0]]))
    ab = alg.rectify(X, "absolute", 0.0)
    assert np.array_equal(ab, np.array([[0, 1.0, 0.5, 0, 0], [0, 2.0, 3.0, 0, 0]]))
    sig = alg.noise_sigma(X)
    assert (sig > 0).all()
    assert (alg.rectify(X, "negative", 2.0) <= neg).all()
    rng = np.random.default_rng(0)
    noisy = 0.1 * rng.standard_normal((2, 5000))
    assert alg.rectify(noisy, "negative", 0.0).mean() > 0.03 and alg.rectify(noisy, "negative", 3.0).mean() < 0.005           # ohne Schwelle Grundlinie, mit Schwelle kaum


def test_noise_sigma_is_robust_to_spikes():
    rng = np.random.default_rng(1)
    x = 0.1 * rng.standard_normal((1, 20000))
    x[0, ::500] -= 5.0
    assert alg.noise_sigma(x)[0] == pytest.approx(0.1, rel=0.1)


def test_spectrogram_of_a_pure_tone():
    n, window, hop = 2048, 64, 8
    t = np.arange(n)
    x = np.sin(2 * np.pi * 8 * t / window)                                   # 8 volle Perioden je Fenster -> Bin 8
    V, centers = alg.spectrogram(x, window, hop)
    assert V.shape == (window // 2 + 1, 1 + (n - window) // hop) and len(centers) == V.shape[1]
    assert (V.argmax(axis=0) == 8).all() and centers[0] == window // 2 and np.diff(centers).tolist() == [hop] * (len(centers) - 1)
    quiet, _ = alg.spectrogram(np.zeros(256), window, hop)
    assert (quiet == 0).all()


def test_frames_to_samples_interpolates_linearly():
    H = np.array([[0.0, 1.0, 0.0]])
    out = alg.frames_to_samples(H, np.array([10, 20, 30]), 41)
    assert out.shape == (1, 41) and out[0, 20] == 1.0 and out[0, 15] == pytest.approx(0.5) and out[0, 0] == 0.0 and out[0, 40] == 0.0


def test_choose_k_hand_instances():
    curve = {1: 1.0, 2: 0.5, 3: 0.1, 4: 0.09, 5: 0.085, 6: 0.08}
    assert alg.choose_k(curve) == 3
    assert alg.choose_k({1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.6}) == 5             # nie ein Knick: die größte geprüfte Zahl
    assert alg.choose_k({1: 1.0, 2: 0.2, 3: 0.19, 4: 0.18}) == 2


def test_error_curve_falls_with_k_on_a_rank_three_instance():
    _, _, V = _instance(k=3)
    curve = alg.error_curve(V, 5, "nndsvd", 0, 200)
    assert curve[1] > curve[2] > curve[3] and curve[3] < 0.05
    assert alg.choose_k(curve) == 3
