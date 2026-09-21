"""Nicht-negative Matrixfaktorisierung (NMF, numpy von Grund auf): multiplikative Updates nach Lee und Seung für den Frobenius-Fehler, Initialisierung zufällig oder per NNDSVD, Gleichrichtung bipolarer Signale
zu einer nicht-negativen Matrix, Betragsspektrogramm für den Einkanal-Fall und die Wahl der Komponentenzahl über die Fehlerkurve."""

from dataclasses import dataclass

import numpy as np

import nm_constants as C

EPS = 1e-12
NOISE_FLOOR_FRACTION = 0.002       # Rausch-Schätzung mindestens 0.2 % der größten Auslenkung (sonst wäre sie bei rauschfreien Daten 0)
RECTIFY_MODES = ("negative", "absolute")
RECTIFY_LABELS = {"negative": "nur die negative Spitze", "absolute": "Betrag (beide Phasen)"}
INITS = ("nndsvd", "random")
INIT_LABELS = {"nndsvd": "NNDSVD (deterministisch)", "random": "zufällig (Neustarts)"}


# --- Nicht-negative Darstellung -----------------------------------------------------------------------------------------------------------


def noise_sigma(X):
    """Robuste Rausch-Standardabweichung je Elektrode: MAD / 0.6745 (Spitzen sind selten)."""
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1) / 0.6745
    return np.maximum(np.maximum(mad, NOISE_FLOOR_FRACTION * np.abs(X - med).max(axis=1)), 1e-12)


def rectify(X, mode="negative", threshold=0.0):
    """Bipolare Elektrodensignale -> V >= 0: "negative" = max(-(x - Median) - c*sigma, 0) (nur die negative Spitze), "absolute" = max(|x - Median| - c*sigma, 0) (beide Phasen);
    c = `threshold` in Rausch-Standardabweichungen (0 = keine Schwelle: das Rauschen wird mit gleichgerichtet und ergibt eine positive Grundlinie)."""
    D = X - np.median(X, axis=1, keepdims=True)
    lift = threshold * noise_sigma(X)[:, None]
    return np.maximum((-D if mode == "negative" else np.abs(D)) - lift, 0.0)


def spectrogram(x, window=64, hop=8):
    """Betrag der Kurzzeit-Fouriertransformation (Hann-Fenster) eines Signals. Rückgabe (V (window/2 + 1, Rahmen), Rahmenmitten in Abtastwerten)."""
    x = np.asarray(x, dtype=float) - np.median(x)
    n_frames = 1 + (len(x) - window) // hop
    idx = np.arange(window)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(window)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)).T, hop * np.arange(n_frames) + window // 2


def frames_to_samples(H, centers, n_samples):
    """Aktivitäten je Rahmen (k, Rahmen) auf das Abtastraster (k, n_samples) abbilden (lineare Interpolation zwischen den Rahmenmitten)."""
    t = np.arange(n_samples)
    return np.array([np.interp(t, centers, h) for h in H])


# --- NMF ------------------------------------------------------------------------------------------------------------------------------


def nndsvd(V, k, fill_mean=True):
    """NNDSVD-Start (Boutsidis und Gallopoulos): aus den führenden Singulärvektoren, jeweils der Teil (positiv oder negativ) mit größerem Norm-Produkt. `fill_mean` ersetzt Nullen durch den Mittelwert von V
    (sonst bleiben Nullen bei multiplikativen Updates ewig Null). Deterministisch. Rückgabe (W (F, k), H (k, T))."""
    F, T = V.shape
    U, S, Vt = np.linalg.svd(V, full_matrices=False)
    W = np.zeros((F, k))
    H = np.zeros((k, T))
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0] = np.sqrt(S[0]) * np.abs(Vt[0])
    for j in range(1, min(k, len(S))):
        x, y = U[:, j], Vt[j]
        xp, xn = np.maximum(x, 0), np.maximum(-x, 0)
        yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
        np_, nn = np.linalg.norm(xp) * np.linalg.norm(yp), np.linalg.norm(xn) * np.linalg.norm(yn)
        if np_ >= nn:
            u, v, sigma = xp / max(np.linalg.norm(xp), EPS), yp / max(np.linalg.norm(yp), EPS), np_
        else:
            u, v, sigma = xn / max(np.linalg.norm(xn), EPS), yn / max(np.linalg.norm(yn), EPS), nn
        scale = np.sqrt(S[j] * sigma)
        W[:, j], H[j] = scale * u, scale * v
    if fill_mean:
        avg = V.mean()
        W[W < EPS] = avg
        H[H < EPS] = avg
    return W, H


def random_start(V, k, seed):
    """Positive Zufallsmatrizen, so skaliert, dass W·H im Mittel V entspricht."""
    rng = np.random.default_rng([seed, 4242])
    scale = np.sqrt(max(V.mean(), EPS) / k)
    return scale * rng.uniform(0.1, 1.0, (V.shape[0], k)) * 2, scale * rng.uniform(0.1, 1.0, (k, V.shape[1])) * 2


def relative_error(V, W, H):
    return float(np.linalg.norm(V - W @ H) / max(np.linalg.norm(V), EPS))


@dataclass(frozen=True)
class NMFResult:
    W: np.ndarray                 # (F, k) Spalten mit Summe 1
    H: np.ndarray                 # (k, T)
    error: float                  # relativer Frobenius-Fehler ||V - WH|| / ||V||
    errors: tuple                 # Fehler vor dem ersten Update und nach jeder Iteration
    snapshots: dict               # Iteration -> (W, H) für ausgewählte Iterationen (Schritt-Ansicht)
    n_iter: int
    converged: bool
    init: str


SNAPSHOT_ITERATIONS = (0, 1, 5, 20, 100)


def _normalise(W, H):
    s = W.sum(axis=0, keepdims=True)
    s = np.where(s < EPS, 1.0, s)
    return W / s, H * s.T


def nmf_once(V, k, init="nndsvd", seed=0, max_iter=C.NMF_MAX_ITER, tol=C.NMF_TOL, sparsity=0.0):
    """Ein NMF-Lauf mit multiplikativen Updates H <- H * (W'V) / (W'WH + lam), W <- W * (VH') / (WHH') (Frobenius). `sparsity` s bestraft die Summe von H (L1): lam = s * Mittelwert(V);
    s = 0 ist die klassische NMF. Hält bis relative Fehleränderung < tol oder `max_iter`."""
    W, H = nndsvd(V, k) if init == "nndsvd" else random_start(V, k, seed)
    W, H = _normalise(W, H)
    lam = sparsity * float(V.mean())
    errors = [relative_error(V, W, H)]
    norm_v2 = max(float((V * V).sum()), EPS)
    snaps = {0: (W.copy(), H.copy())}
    converged = False
    for it in range(1, max_iter + 1):
        H = H * (W.T @ V) / (W.T @ W @ H + lam + EPS)
        VHt, HHt = V @ H.T, H @ H.T
        W = W * VHt / (W @ HHt + EPS)
        # ||V - WH||^2 = ||V||^2 - 2 tr(W' V H') + tr(W'W HH'): nur n x k und k x k statt der n x T Matrix W·H (vor der Normierung, die das Produkt nicht ändert)
        err2 = norm_v2 - 2.0 * float((W * VHt).sum()) + float(((W.T @ W) * HHt).sum())
        W, H = _normalise(W, H)
        errors.append(float(np.sqrt(max(err2, 0.0) / norm_v2)))
        if it in SNAPSHOT_ITERATIONS:
            snaps[it] = (W.copy(), H.copy())
        if errors[-2] - errors[-1] < tol * max(errors[-2], EPS):
            converged = True
            break
    snaps[it] = (W.copy(), H.copy())
    errors[-1] = relative_error(V, W, H)                 # der letzte Wert exakt (die Spurformel verliert bei sehr kleinen Fehlern Stellen)
    return NMFResult(W, H, errors[-1], tuple(errors), snaps, it, converged, init)


def nmf(V, k, init="nndsvd", seed=0, n_restarts=C.NMF_RESTARTS, max_iter=C.NMF_MAX_ITER, tol=C.NMF_TOL, sparsity=0.0):
    """NMF mit Neustarts (nur bei zufälligem Start; NNDSVD ist deterministisch): der Lauf mit dem kleinsten Fehler zählt. Rückgabe (bestes Ergebnis, Fehler aller Läufe)."""
    runs = [nmf_once(V, k, init, seed, max_iter, tol, sparsity)] if init == "nndsvd" else [nmf_once(V, k, init, seed + r, max_iter, tol, sparsity) for r in range(n_restarts)]
    best = min(runs, key=lambda r: r.error)
    return best, tuple(r.error for r in runs)


# --- Komponentenzahl ---------------------------------------------------------------------------------------------------------------------


def error_curve(V, k_max, init="nndsvd", seed=0, max_iter=C.NMF_MAX_ITER, sparsity=0.0):
    """Relativer Fehler je Komponentenzahl k = 1 ... k_max (jeweils ein NMF-Lauf mit dem gewählten Start, bei zufälligem Start der beste von drei)."""
    return {k: nmf(V, k, init, seed, 3, max_iter, C.NMF_TOL, sparsity)[0].error for k in range(1, k_max + 1)}


def choose_k(curve, ratio=C.RANK_RATIO):
    """Knick der Fehlerkurve: die kleinste Komponentenzahl k >= 2, bei der ein weiteres Element weniger als `ratio` mal so viel Fehler spart wie das letzte hinzugefügte (Gewinn(k -> k+1) < ratio * Gewinn(k-1 -> k)).
    Sinkt der Fehler nie so stark, gilt die größte geprüfte Zahl."""
    ks = sorted(curve)
    gain = {k: curve[k] - curve[k + 1] for k in ks[:-1]}
    for k in ks[1:-1]:
        if gain[k] < ratio * gain[k - 1]:
            return k
    return ks[-1]
