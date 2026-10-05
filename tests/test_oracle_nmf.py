"""Orakel (anderer Rechenweg): scikit-learn (multiplikative Updates, NNDSVD), Schleifenimplementierung der Updates inkl. L1-Strafe, explizite DFT,
SciPy (Zuordnung, NNLS, MAD, Interpolation) und eine Schleifen-Nachbildung des Szenarios. Kleine Instanzen, wenige Sekunden."""

import math
import warnings

import numpy as np
import pytest

import nm_algorithm as alg
import nm_constants as C
import nm_evaluation as ev
import nm_scenario as sc

sk_nmf = pytest.importorskip("sklearn.decomposition")
sp_opt = pytest.importorskip("scipy.optimize")
sp_stats = pytest.importorskip("scipy.stats")
sp_interp = pytest.importorskip("scipy.interpolate")


def _fixed_start(monkeypatch, W0, H0):
    monkeypatch.setattr(alg, "random_start", lambda V, k, seed: (W0.copy(), H0.copy()))


@pytest.mark.parametrize("seed", range(6))
def test_multiplicative_updates_equal_scikit_learn_with_the_same_start(monkeypatch, seed):
    """Dieselbe Update-Reihenfolge (erst H, dann W) wie die Demo: scikit-learn auf V^T (dort kommt "W" = unser H zuerst) mit identischem Start.
    Die Spaltennormierung ändert das Produkt W·H nicht, also stimmen die Produkte überein."""
    rng = np.random.default_rng(seed)
    F, T, k = int(rng.integers(3, 9)), int(rng.integers(8, 40)), int(rng.integers(1, 4))
    V = rng.random((F, T)) * (rng.random((F, T)) > 0.3)
    V[0, 0] += 0.5
    its = int(rng.integers(3, 40))
    W0, H0 = alg.random_start(V, k, seed)
    Wn, Hn = alg._normalise(W0, H0)
    _fixed_start(monkeypatch, W0, H0)
    res = alg.nmf_once(V, k, "random", 0, max_iter=its, tol=-1.0)
    sk = sk_nmf.NMF(n_components=k, init="custom", solver="mu", beta_loss="frobenius", tol=0, max_iter=its, alpha_W=0, alpha_H=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ht = sk.fit_transform(V.T, W=Hn.T.copy(), H=Wn.T.copy())
    assert np.allclose((Ht @ sk.components_).T, res.W @ res.H, atol=1e-8 * max(V.max(), 1.0))
    assert res.error == pytest.approx(np.linalg.norm(V - res.W @ res.H) / np.linalg.norm(V), abs=1e-9)


def _mu_by_loops(V, W, H, lam, its):
    F, k = W.shape
    T = H.shape[1]
    for _ in range(its):
        WH = W @ H
        Hn = H.copy()
        for i in range(k):
            for j in range(T):
                Hn[i, j] = H[i, j] * sum(W[f, i] * V[f, j] for f in range(F)) / (sum(W[f, i] * WH[f, j] for f in range(F)) + lam + alg.EPS)
        H = Hn
        WH = W @ H
        Wn = W.copy()
        for f in range(F):
            for i in range(k):
                Wn[f, i] = W[f, i] * sum(V[f, j] * H[i, j] for j in range(T)) / (sum(WH[f, j] * H[i, j] for j in range(T)) + alg.EPS)
        W = Wn
        s = W.sum(axis=0)
        W, H = W / s, H * s[:, None]
    return W, H


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 3.0])
def test_updates_with_l1_penalty_equal_an_elementwise_loop_implementation(monkeypatch, sparsity):
    rng = np.random.default_rng(3)
    V = rng.random((4, 9))
    W0, H0 = alg.random_start(V, 3, 1)
    Wn, Hn = alg._normalise(W0, H0)
    _fixed_start(monkeypatch, W0, H0)
    res = alg.nmf_once(V, 3, "random", 0, max_iter=12, tol=-1.0, sparsity=sparsity)
    Wo, Ho = _mu_by_loops(V, Wn, Hn, sparsity * V.mean(), 12)
    assert np.allclose(res.W, Wo, rtol=1e-8, atol=1e-10) and np.allclose(res.H, Ho, rtol=1e-8, atol=1e-10)


def test_trace_formula_errors_equal_the_direct_definition_at_every_snapshot():
    rng = np.random.default_rng(2)
    V = rng.random((6, 50)) * (rng.random((6, 50)) > 0.5)
    V[1, 1] += 1
    res = alg.nmf_once(V, 3, "random", 4, max_iter=100, tol=-1.0)
    for it, (W, H) in res.snapshots.items():
        if it < res.n_iter:
            assert res.errors[it] == pytest.approx(np.linalg.norm(V - W @ H) / np.linalg.norm(V), abs=1e-6)


def test_nndsvd_equals_scikit_learn_initialisation():
    nmf_mod = pytest.importorskip("sklearn.decomposition._nmf")
    if not hasattr(nmf_mod, "_initialize_nmf"):
        pytest.skip("scikit-learn ohne _initialize_nmf")
    rng = np.random.default_rng(9)
    for k in (1, 2, 3, 4):
        V = rng.random((7, 30)) * (rng.random((7, 30)) > 0.4)
        V[0, 0] += 1
        V[1, 1] += 1
        W, H = alg.nndsvd(V, k, fill_mean=False)
        Ws, Hs = nmf_mod._initialize_nmf(V, k, init="nndsvd", random_state=0)
        assert np.allclose(W, Ws, atol=1e-6) and np.allclose(H, Hs, atol=1e-6)


def test_rectify_and_noise_sigma_equal_scipy_mad_and_an_elementwise_loop():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3, 400)) * np.array([[0.2], [1.0], [0.5]]) + rng.normal(size=(3, 1))
    X[:, ::40] -= 4
    ref_sigma = sp_stats.median_abs_deviation(X, axis=1, scale="normal")
    assert np.allclose(alg.noise_sigma(X), ref_sigma, rtol=1e-3)
    for mode in alg.RECTIFY_MODES:
        for thr in (0.0, 2.0):
            V = alg.rectify(X, mode, thr)
            for e in range(3):
                med, s = np.median(X[e]), alg.noise_sigma(X)[e]
                for i in range(0, 400, 7):
                    d = X[e, i] - med
                    assert V[e, i] == pytest.approx(max((-d if mode == "negative" else abs(d)) - thr * s, 0.0), abs=1e-12)


def test_spectrogram_equals_an_explicit_dft_with_a_hann_window():
    rng = np.random.default_rng(4)
    for w, hop in ((16, 4), (32, 4), (8, 3)):
        x = rng.standard_normal(150) + 2.0
        V, centers = alg.spectrogram(x, w, hop)
        xm = x - np.median(x)
        hann = [0.5 - 0.5 * math.cos(2 * math.pi * i / (w - 1)) for i in range(w)]
        nf = 1 + (150 - w) // hop
        for f in range(0, nf, 5):
            seg = [xm[f * hop + i] * hann[i] for i in range(w)]
            for kk in range(0, w // 2 + 1, 3):
                z = sum(seg[i] * complex(math.cos(2 * math.pi * kk * i / w), -math.sin(2 * math.pi * kk * i / w)) for i in range(w))
                assert V[kk, f] == pytest.approx(abs(z), abs=1e-9)
        assert np.array_equal(centers, hop * np.arange(nf) + w // 2)


def test_frames_to_samples_equals_scipy_interp1d_with_edge_hold():
    rng = np.random.default_rng(6)
    H = rng.random((3, 20))
    cen = np.cumsum(rng.integers(1, 5, 20)) + 5
    out = alg.frames_to_samples(H, cen, int(cen[-1] + 10))
    f = sp_interp.interp1d(cen, H, axis=1, bounds_error=False, fill_value=(H[:, 0], H[:, -1]))
    assert np.allclose(out, f(np.arange(out.shape[1])))


def test_mixing_cos_equals_scipy_assignment_and_column_similarity_equals_loops():
    rng = np.random.default_rng(8)
    for _ in range(40):
        n, m, mh = int(rng.integers(1, 7)), int(rng.integers(2, 6)), int(rng.integers(1, 6))
        A = np.abs(rng.standard_normal((n, m))) + 0.01
        Ah = np.abs(rng.standard_normal((n, mh))) + 0.01
        cm = np.abs((A / np.linalg.norm(A, axis=0)).T @ (Ah / np.linalg.norm(Ah, axis=0)))
        r, c = sp_opt.linear_sum_assignment(cm, maximize=True)
        assert ev.mixing_cos(A, Ah) == pytest.approx(cm[r, c].sum() / m, abs=1e-9)
        cos = [A[:, i] @ A[:, j] / np.linalg.norm(A[:, i]) / np.linalg.norm(A[:, j]) for i in range(m) for j in range(m) if i != j]
        assert ev.column_similarity(A) == pytest.approx(max(cos), abs=1e-12)


def test_true_factorisation_error_is_close_to_the_exact_nnls_optimum_but_never_below_it():
    """Referenz "Fehler mit den wahren Mischspalten": multiplikative Updates mit festem W gegen die exakte Lösung (NNLS je Spalte)."""
    ds = ev.make_dataset(seed=100000)
    V = alg.rectify(ds.X, "negative", 2.0)
    W = ds.A / ds.A.sum(axis=0, keepdims=True)
    H = np.column_stack([sp_opt.nnls(W, V[:, j])[0] for j in range(V.shape[1])])
    opt = np.linalg.norm(V - W @ H) / np.linalg.norm(V)
    got = ev.true_factorisation_error(V, ds.A)
    assert opt - 1e-9 <= got <= opt + 0.01


@pytest.mark.parametrize("m,n,syn,sim,jit", [(4, 4, 0.0, 1.0, 0.0), (3, 2, 0.9, 2.0, 0.3), (5, 8, 0.3, 0.0, 0.1)])
def test_dataset_equals_a_loop_rebuild_including_synchrony_similarity_and_jitter(m, n, syn, sim, jit):
    T, seed = 5000, 21
    ds = sc.make_dataset(m, n, 1.0, sim, jit, 0.0, T, seed, syn)
    pos = [0.5] if n == 1 else list(np.linspace(0, 1, n))
    A = np.zeros((n, m))
    for i in range(m):
        d = np.array([math.hypot(p - C.NEURON_POSITIONS[i][0], C.NEURON_POSITIONS[i][1]) for p in pos])
        a = 1.0 / (d ** 2 + C.DISTANCE_EPS)
        A[:, i] = C.NEURON_AMPLITUDES[i] * a / a.max()
    assert np.allclose(ds.A, A)
    for i in range(m):
        starts = sc.synchronous_times(i, T, seed, 1.0, syn)
        assert len(starts) < 2 or np.diff(starts).min() >= C.REFRACTORY
        sig = C.SIGMA_CENTER + sim * (C.NEURON_SIGMAS[i] - C.SIGMA_CENTER)
        w = [-math.exp(-((k - C.PEAK_INDEX) / sig) ** 2) + C.OVERSHOOT * math.exp(-((k - (C.PEAK_INDEX + 2.5 * sig)) / (1.6 * sig)) ** 2) for k in range(C.WAVEFORM_LENGTH)]
        amps = np.ones(len(starts)) if jit == 0 else 1.0 + jit * np.random.default_rng([seed, 3000 + i]).standard_normal(len(starts))
        s = np.zeros(T)
        for a, t0 in zip(amps, starts):
            for k in range(C.WAVEFORM_LENGTH):
                if t0 + k < T:
                    s[t0 + k] += a * w[k]
        assert np.allclose((s - s.mean()) / s.std(), ds.S[i], atol=1e-9)
        assert np.array_equal(ds.spike_times[i], starts + int(np.argmin(w)))


def test_synchrony_share_matches_its_definition():
    """Anteil der Spikes von Neuron 1, den die anderen Neuronen übernehmen, entspricht dem Regler (bis auf Refraktär-Verluste)."""
    for syn in (0.3, 0.9):
        shares = []
        for seed in range(100000, 100003):
            s0 = sc.synchronous_times(0, 20000, seed, 1.0, syn)
            for i in (1, 2):
                other = set(sc.synchronous_times(i, 20000, seed, 1.0, syn).tolist())
                shares.append(np.mean([x in other for x in s0]))
        assert np.mean(shares) == pytest.approx(syn, abs=0.04)


def test_fastica_copy_equals_scikit_learn_with_the_same_start():
    import nm_ica as ica
    rng = np.random.default_rng(12)
    X = rng.standard_normal((4, 4)) @ rng.laplace(size=(4, 2500))
    ours = ica.fit_ica(X, 4, "logcosh", "symmetric", init_start=3, max_iter=500, tol=1e-9)
    W0 = np.random.default_rng([3, 4242]).standard_normal((4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sk_nmf.FastICA(algorithm="parallel", whiten=False, fun="logcosh", w_init=W0, tol=1e-9, max_iter=500).fit(ours.whitening.Z.T)
    assert np.allclose(ours.W, ref.components_, atol=1e-8)
