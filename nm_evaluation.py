"""Auswertung der NMF-Demo: die Spitzen-Zuordnung und -Treffer der Vorgänger (aus ica/sobi/sca-demo übernommen), die Analyse einer Aufnahme für NMF, ICA, SOBI, SCA und die Referenzen
(beste Einzelelektrode, Schwelle), die Mischspalten-Winkel, Sweeps, Rang- und Initialisierungs-Tabellen, Szenen und das Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import nm_algorithm as alg
import nm_constants as C
import nm_ica as ica
import nm_scenario as sc
import nm_sca as sca
import nm_sobi as sobi


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def assign(C_abs):
    """Optimale Zuordnung Quelle -> Schätzung (jede Schätzung höchstens einer Quelle), Summe der |Korrelationen| maximal. Exakt per Bitmasken-DP.
    Rückgabe: Liste je Quelle mit dem Index der Schätzung oder -1 (nicht zugeordnet, nur wenn es weniger Schätzungen als Quellen gibt)."""
    k, nc = C_abs.shape
    best = {}

    def solve(row, used):
        if row == k:
            return 0.0, ()
        key = (row, used)
        if key in best:
            return best[key]
        result = (solve(row + 1, used)[0], (-1,) + solve(row + 1, used)[1])
        for j in range(nc):
            if not used >> j & 1:
                value, rest = solve(row + 1, used | 1 << j)
                if value + C_abs[row, j] > result[0] + 1e-12:
                    result = (value + C_abs[row, j], (j,) + rest)
        best[key] = result
        return result

    return list(solve(0, 0)[1])


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def sir_db(corr):
    """Signal-zu-Interferenz in dB aus der Korrelation: rho^2 / (1 - rho^2)."""
    r2 = np.clip(np.asarray(corr) ** 2, 1e-9, 1.0 - 1e-9)
    return 10.0 * np.log10(r2 / (1.0 - r2))


def amari_index(P):
    """Amari-Index einer (nc, k)-Matrix P = Entmischung x Mischung; 0 = perfekt (nur Permutation und Skalierung), 1 = schlechtestmöglich; verallgemeinert auf nicht quadratische P."""
    P = np.abs(P)
    k = P.shape[1]
    rows = (P.sum(axis=1) / P.max(axis=1) - 1.0).sum()
    cols = (P.sum(axis=0) / P.max(axis=0) - 1.0).sum()
    d = max(k, P.shape[0])
    return float((rows + cols) / (2.0 * d * (d - 1))) if d > 1 else 0.0


# --- Spike-Erkennung auf den Schätzquellen ------------------------------------------------------------------------------------
DETECT_MIN_SEPARATION = 15         # Abtastwerte zwischen zwei erkannten Spitzen
DETECT_TOLERANCE = 4               # erlaubte Abweichung zur wahren Spitze
DETECT_SIGMA_FACTOR = 4.0          # Schwelle: 4 robuste Standardabweichungen (MAD) ...
DETECT_DEPTH_FRACTION = 0.3        # ... mindestens aber 30 % der typischen Spitzentiefe (sonst würde ein rauschfreies Signal jeden Rest melden)


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie beim Neuron): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe)."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -DETECT_SIGMA_FACTOR * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - DETECT_MIN_SEPARATION): t + DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen
            threshold = min(threshold, DETECT_DEPTH_FRACTION * float(np.median(x[peaks])))
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= DETECT_TOLERANCE and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)



def excess_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = (x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, keepdims=True)
    return (x ** 4).mean(axis=-1) - 3.0


# --- Daten, Einstellungen, Mischspalten --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    mode: str = "negative"                   # Gleichrichtung: "negative" oder "absolute"
    threshold: float = 2.0                   # Rauschschwelle der Gleichrichtung in Rausch-Standardabweichungen
    k_mode: str = "known"                    # Komponentenzahl: "known" (= Neuronenzahl) oder "unknown" (per Knick der Fehlerkurve)
    init: str = "random"                     # "random" (Neustarts) oder "nndsvd"
    sparsity: float = 0.0                    # L1-Strafe auf H (0 = klassische NMF)
    max_iter: int = C.NMF_MAX_ITER
    contrast: str = C.DEFAULT_CONTRAST       # ICA-Vergleich
    init_start: int = C.DEFAULT_INIT_START   # Start der NMF-Zufallsinitialisierung und der ICA


DATA_KEYS = ("m", "n", "rate_scale", "similarity", "jitter", "noise", "n_samples", "synchrony")
DEFAULT_DATA = dict(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, rate_scale=C.DEFAULT_RATE_SCALE, similarity=C.DEFAULT_SIMILARITY, jitter=C.DEFAULT_JITTER, noise=C.DEFAULT_NOISE,
                    n_samples=C.DEFAULT_N_SAMPLES, synchrony=C.DEFAULT_SYNCHRONY)


def make_dataset(m=C.DEFAULT_N_NEURONS, n=C.DEFAULT_N_ELECTRODES, rate_scale=C.DEFAULT_RATE_SCALE, similarity=C.DEFAULT_SIMILARITY, jitter=C.DEFAULT_JITTER, noise=C.DEFAULT_NOISE,
                 n_samples=C.DEFAULT_N_SAMPLES, synchrony=C.DEFAULT_SYNCHRONY, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, seed, synchrony)


def n_components(ds):
    """Zahl der Quellen für ICA und SOBI: die der Neuronen, höchstens die der Elektroden."""
    return min(ds.n_electrodes, ds.S.shape[0])


def mixing_cos(A_true, A_hat):
    """Mittlere |Kosinus-Ähnlichkeit| zwischen wahren und geschätzten Mischspalten nach optimaler Zuordnung (Neuronen ohne Partner zählen 0). 1 = alle Mischspalten richtig."""
    A = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
    B = A_hat / np.maximum(np.linalg.norm(A_hat, axis=0, keepdims=True), 1e-12)
    cm = np.abs(A.T @ B)
    idx = assign(cm)
    return float(np.mean([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)]))


def column_similarity(A):
    """Größte Kosinus-Ähnlichkeit zweier verschiedener wahrer Mischspalten (nahe 1 = die Spalten liegen fast aufeinander; dann ist die Zerlegung schwer)."""
    B = A / np.linalg.norm(A, axis=0, keepdims=True)
    G = B.T @ B
    np.fill_diagonal(G, 0.0)
    return float(G.max()) if G.size > 1 else 0.0


def source_scores(ds, sources):
    """Spitzen-F1 und Neuron-Korrelation (Mittel über die Neuronen) einer Schätzung: Zuordnung per Korrelation, Vorzeichen und Skala per Regression, Spitzen auf der Spur - für alle Verfahren gleich."""
    m = ds.n_neurons
    idx, corr, aligned = matched(ds.S, sources)
    return float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)])), float(corr.mean())


# --- Analyse einer Aufnahme --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    params: tuple                 # (m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony, seed)
    single: bool                  # eine Elektrode: NMF auf dem Spektrogramm statt auf den gleichgerichteten Elektroden
    V: np.ndarray                 # die nicht-negative Matrix (Elektroden × Zeit oder Frequenz × Rahmen)
    centers: np.ndarray           # Rahmenmitten (nur Einkanal)
    nmf: alg.NMFResult
    k: int                        # verwendete Komponentenzahl
    curve: dict                   # Fehlerkurve über k (nur bei unbekannter Komponentenzahl, sonst leer)
    start_errors: tuple           # Fehler aller Neustarts
    sources: np.ndarray           # (k, T) NMF-Aktivitäten auf dem Abtastraster
    scores: dict                  # Methode -> {"f1", "corr", "angle"} (angle nur bei mehreren Elektroden)
    error_true: float             # Fehler von V gegen A_wahr·H (H frei angepasst) - der Vergleichswert zum NMF-Fehler (nur Array)
    baseline_f1: float            # Schwelle auf der stärksten Elektrode: alle erkannten Spitzen jedem Neuron zugerechnet
    column_similarity: float
    seconds: dict


def prepare(ds, settings):
    """Die nicht-negative Matrix: gleichgerichtete Elektroden (mehrere Elektroden) oder Betragsspektrogramm der Elektrode (eine)."""
    if ds.n_electrodes == 1:
        V, centers = alg.spectrogram(ds.X[0], C.SPECTROGRAM_WINDOW, C.SPECTROGRAM_HOP)
        return V, centers
    return alg.rectify(ds.X, settings.mode, settings.threshold), np.zeros(0)


def true_factorisation_error(V, A, iterations=300):
    """Fehler von V gegen die wahre Mischung: W = A (Spalten mit Summe 1), H so angepasst (multiplikative Updates, W fest) - was die Zerlegung mit den wahren Mischspalten erreichen würde."""
    W = A / A.sum(axis=0, keepdims=True)
    H = np.full((W.shape[1], V.shape[1]), float(V.mean()))
    for _ in range(iterations):
        H = H * (W.T @ V) / (W.T @ W @ H + alg.EPS)
    return alg.relative_error(V, W, H)


def analyse(ds, settings, params=None, comparators=True):
    secs = {}
    m = ds.n_neurons
    single = ds.n_electrodes == 1
    V, centers = prepare(ds, settings)
    t0 = time.perf_counter()
    curve = {}
    k = m
    if settings.k_mode == "unknown":
        curve = alg.error_curve(V, min(C.NMF_K_MAX, max(V.shape[0], 2) + 4 if not single else C.NMF_K_MAX), settings.init, settings.init_start, 100, settings.sparsity)
        k = alg.choose_k(curve)
    res, start_errors = alg.nmf(V, k, settings.init, settings.init_start, C.NMF_RESTARTS, settings.max_iter, C.NMF_TOL, settings.sparsity)
    secs["nmf"] = time.perf_counter() - t0
    sources = alg.frames_to_samples(res.H, centers, ds.X.shape[1]) if single else res.H
    f1, corr = source_scores(ds, sources)
    scores = {"nmf": {"f1": f1, "corr": corr, "angle": float("nan") if single else mixing_cos(ds.A, res.W)}}
    if comparators:
        nc = n_components(ds)
        t0 = time.perf_counter()
        model = ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start)
        secs["ica"] = time.perf_counter() - t0
        estimates = {"ica": (model.sources, np.linalg.pinv(model.unmixing))}
        t0 = time.perf_counter()
        sm = sobi.fit_sobi(ds.X, nc, C.SOBI_LAGS)
        secs["sobi"] = time.perf_counter() - t0
        estimates["sobi"] = (sm.sources, np.linalg.pinv(sm.unmixing))
        t0 = time.perf_counter()
        cm = sca.fit_sca(ds.X, m, "l1", seed=settings.init_start)
        secs["sca"] = time.perf_counter() - t0
        estimates["sca"] = (cm.sources, cm.A_hat)
        estimates["pca"] = (ica.pca_components(ds.X, nc), None)
        estimates["electrode"] = (ds.X, None)
        for name, (est, A_hat) in estimates.items():
            f, c = source_scores(ds, est)
            scores[name] = {"f1": f, "corr": c, "angle": mixing_cos(ds.A, A_hat) if (A_hat is not None and not single) else float("nan")}
    strongest = int(np.argmax(np.abs(ds.X - np.median(ds.X, axis=1, keepdims=True)).max(axis=1)))
    det = detect_spikes(ds.X[strongest] - np.median(ds.X[strongest]))
    baseline = float(np.mean([spike_f1(det, ds.spike_times[i]) for i in range(m)]))
    error_true = float("nan") if single else true_factorisation_error(V, ds.A)
    return Analysis(ds, settings, params, single, V, centers, res, k, curve, start_errors, sources, scores, error_true, baseline, column_similarity(ds.A), secs)


def analyse_for(params, settings, comparators=True):
    """`params` = (m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony, seed)."""
    return analyse(make_dataset(*params), settings, params, comparators)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n": (1, 2, 3, 4, 6, 8),
    "m": (2, 3, 4, 5),
    "noise": (0.05, 0.2, 0.5, 1.0),
    "rate_scale": (0.25, 0.5, 1.0, 2.0),
    "similarity": (0.0, 0.5, 1.0, 1.5, 2.0),
    "jitter": (0.0, 0.1, 0.2, 0.3),
    "synchrony": (0.0, 0.3, 0.6, 0.9),
    "threshold": (0.0, 1.0, 2.0, 3.0),
    "sparsity": (0.0, 0.5, 1.0, 2.0, 3.0),
    "max_iter": (20, 100, 300, 1000),
}
SWEEP_LABELS = {"n": "Anzahl Elektroden (1 = Einkanal-Spektrogramm)", "m": "Anzahl Neuronen", "noise": "Rauschen (relativ zum Neuronen-Signal)", "rate_scale": "Feuerrate (Faktor)",
                "similarity": "Wellenform-Ähnlichkeit (0 = alle gleich, 2 = Breitenunterschiede doppelt)", "jitter": "Amplitudenschwankung", "synchrony": "Synchronität (Anteil gemeinsamer Spikes)",
                "threshold": "Rauschschwelle der Gleichrichtung (Rausch-σ)", "sparsity": "Sparsität λ (L1-Strafe auf H)", "max_iter": "Iterationen"}
SETTING_PARAMETERS = ("threshold", "sparsity", "max_iter")
METHODS = ("nmf", "ica", "sobi", "sca")


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r[key] for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def _record(a):
    out = {name: a.scores[name]["f1"] for name in a.scores}
    out.update({name + "_corr": a.scores[name]["corr"] for name in a.scores})
    out.update({"nmf_angle": a.scores["nmf"]["angle"], "nmf_error": a.nmf.error, "error_true": a.error_true, "baseline": a.baseline_f1, "k": float(a.k),
                "column_similarity": a.column_similarity})
    return out


def sweep(parameter, values=None, settings=Settings(), comparators=True, **base):
    """Mittel, Streuung und Spanne über die festen Sweep-Datensätze der Kennzahlen von NMF und (bei Reglern der Daten) ICA, SOBI, SCA in Abhängigkeit von einem Regler."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            kw = {**DEFAULT_DATA, **base}
            if parameter in SETTING_PARAMETERS:
                s = Settings(**{**settings.__dict__, parameter: x})
                per_seed.append(_record(analyse(make_dataset(seed=seed, **kw), s, comparators=False)))
            else:
                kw[parameter] = x
                per_seed.append(_record(analyse(make_dataset(seed=seed, **kw), settings, comparators=comparators)))
        rows.append(_summarise(x, per_seed))
    return rows


def rank_table(settings=Settings(), neurons=(2, 3, 4, 5), n=6, **base):
    """Komponentenzahl unbekannt: mittlere Fehlerkurve (k = 1 ... 8) und die per Knick gewählten k je wahrer Neuronenzahl über die Sweep-Datensätze."""
    rows = []
    for m in neurons:
        curves, chosen = [], []
        for seed in C.SWEEP_SEEDS:
            kw = {**DEFAULT_DATA, **base, "m": m, "n": n}
            ds = make_dataset(seed=seed, **kw)
            V, _ = prepare(ds, settings)
            curve = alg.error_curve(V, C.NMF_K_MAX, settings.init, settings.init_start, 100, settings.sparsity)
            curves.append([curve[k] for k in range(1, C.NMF_K_MAX + 1)])
            chosen.append(alg.choose_k(curve))
        rows.append({"m": m, "curve": [float(v) for v in np.mean(curves, axis=0)], "chosen": chosen, "correct": float(np.mean([k == m for k in chosen]))})
    return rows


def init_table(settings=Settings(), n_starts=8, **base):
    """Initialisierung: je Sweep-Datensatz acht zufällige Starts (Fehler und Spitzen-F1) und der NNDSVD-Start. Kennzahlen: Streuung des F1 über die Starts, Korrelation Fehler-F1 über alle Starts,
    F1 des Starts mit dem kleinsten Fehler gegen Mittel und Bestwert der Starts, F1 von NNDSVD."""
    per = []
    for seed in C.SWEEP_SEEDS:
        kw = {**DEFAULT_DATA, **base}
        ds = make_dataset(seed=seed, **kw)
        V, centers = prepare(ds, settings)
        runs = []
        for r in range(n_starts):
            res = alg.nmf_once(V, ds.n_neurons, "random", seed=settings.init_start + r, max_iter=settings.max_iter, sparsity=settings.sparsity)
            src = alg.frames_to_samples(res.H, centers, ds.X.shape[1]) if ds.n_electrodes == 1 else res.H
            runs.append((res.error, source_scores(ds, src)[0]))
        nn = alg.nmf_once(V, ds.n_neurons, "nndsvd", max_iter=settings.max_iter, sparsity=settings.sparsity)
        src = alg.frames_to_samples(nn.H, centers, ds.X.shape[1]) if ds.n_electrodes == 1 else nn.H
        errs, f1s = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
        per.append({"runs": [(float(e), float(f)) for e, f in runs], "std": float(f1s.std()), "corr": float(np.corrcoef(errs, f1s)[0, 1]) if f1s.std() > 0 and errs.std() > 0 else float("nan"), "lowest": float(f1s[np.argmin(errs)]),
                    "mean": float(f1s.mean()), "best": float(f1s.max()), "worst": float(f1s.min()), "nndsvd": source_scores(ds, src)[0], "err_mean": float(errs.mean()), "err_nndsvd": nn.error})
    return {k: float(np.nanmean([p[k] for p in per])) for k in per[0] if k != "runs"} | {"per_seed": per}


SCENES = (
    ("Standardfall (4 Neuronen, 4 Elektroden)", dict()),
    ("Synchrones Feuern (90 % gemeinsame Spikes)", dict(synchrony=0.9)),
    ("Starkes Rauschen (1.0)", dict(noise=1.0)),
    ("Zwei Elektroden, vier Neuronen", dict(n=2)),
    ("Fünf Neuronen auf vier Elektroden", dict(m=5)),
    ("Zwei Neuronen, acht Elektroden", dict(m=2, n=8)),
    ("Eine Elektrode: Breiten wie gewohnt (2 Neuronen)", dict(m=2, n=1)),
    ("Eine Elektrode: sehr verschiedene Breiten (2 Neuronen, Ähnlichkeit 2)", dict(m=2, n=1, similarity=2.0)),
)
SCENE_METHODS = ("nmf", "ica", "sobi", "sca")


def scene_table(settings=Settings(), **base):
    """NMF, ICA, SOBI und SCA in acht Szenen (Spitzen-F1, Mittel und Spanne über die Sweep-Datensätze); dazu die Schwelle als Referenz für die Einkanal-Szenen."""
    rows = []
    for label, scene in SCENES:
        kw = {**DEFAULT_DATA, **base, **scene}
        acc = {name: [] for name in SCENE_METHODS}
        acc["baseline"] = []
        acc["angle"] = []
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings, comparators=True)
            for name in SCENE_METHODS:
                acc[name].append(a.scores[name]["f1"])
            acc["baseline"].append(a.baseline_f1)
            acc["angle"].append(a.scores["nmf"]["angle"])
        row = {"scene": label}
        for name, vals in acc.items():
            row[name] = float(np.nanmean(vals)) if not np.isnan(vals).all() else float("nan")
            row[name + "_min"], row[name + "_max"] = (float(np.nanmin(vals)), float(np.nanmax(vals))) if not np.isnan(vals).all() else (float("nan"), float("nan"))
        rows.append(row)
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_MARGIN = 0.08             # F1-Abstand, ab dem ein Sieger benannt wird
VERDICT_DROP = 0.10               # Verlust gegenüber dem Referenzlauf, ab dem eine Ursache benannt wird


def _nmf_f1(params, settings):
    return analyse(make_dataset(*params), settings, comparators=False).scores["nmf"]["f1"]


def verdict(a):
    """(Art, Code, Kennzahlen). Nur mit großer Marge; bei schwachem NMF benennt ein Referenzlauf (nur NMF, ohne die gestörte Eigenschaft) die Ursache."""
    ds, s = a.ds, a.settings
    m, n = ds.n_neurons, ds.n_electrodes
    nmf_f1 = a.scores["nmf"]["f1"]
    others = {k: v["f1"] for k, v in a.scores.items() if k in ("ica", "sobi", "sca")}
    best_name = max(others, key=others.get) if others else ""
    best = others.get(best_name, float("nan"))
    data = {"f1": nmf_f1, "best": best, "best_name": best_name, "corr": a.scores["nmf"]["corr"], "angle": a.scores["nmf"]["angle"], "k": a.k, "m": m, "n": n, "error": a.nmf.error, "error_true": a.error_true,
            "baseline": a.baseline_f1, "column_similarity": a.column_similarity, "electrode_f1": a.scores.get("electrode", {}).get("f1", float("nan")), **{f"{k}_f1": v for k, v in others.items()}}
    if s.k_mode == "unknown" and a.k != m:
        return "warning", "wrong_k", data
    if n == 1:
        if nmf_f1 - a.baseline_f1 >= VERDICT_MARGIN:
            return "success", "single_works", data
        return "warning", "single_weak", data
    if nmf_f1 - best >= VERDICT_MARGIN:
        return "success", "nmf_wins", data
    if best - nmf_f1 >= VERDICT_MARGIN:
        if n < m:
            return "warning", "too_few_electrodes", data
        if a.params is not None:
            base = list(a.params)
            refs = {}
            if base[5] > 0.2:
                refs["noise"] = _nmf_f1(tuple(base[:5] + [C.DEFAULT_NOISE] + base[6:]), s) - nmf_f1
            if refs:
                cause, drop = max(refs.items(), key=lambda kv: kv[1])
                data["drop"] = drop
                if drop > VERDICT_DROP:
                    return "warning", cause, data
        return "warning", "others_win", data
    return "success", "comparable", data


# --- Ansichten für die App ---------------------------------------------------------------------------------------------------------------


def component_view(a):
    """Zuordnung der NMF-Komponenten zu den Neuronen: (Komponente je Neuron, |Korrelation| (m, k), Aktivitäten in Skala und Vorzeichen der wahren Neuronen, erkannte Spitzenzeiten je Neuron)."""
    idx, corr, aligned = matched(a.ds.S, a.sources)
    detected = [detect_spikes(aligned[i]) for i in range(a.ds.n_neurons)]
    return idx, correlation_matrix(a.ds.S, a.sources), aligned, detected


def progress(a):
    """Zwischenstände der Faktorisierung: (Iterationen, Korrelation, Mischspalten-Kosinus, Spitzen-F1) - zeigt, wie sich die Zerlegung der Wahrheit nähert (oder wieder entfernt)."""
    its = sorted(a.nmf.snapshots)
    corr, cos, f1 = [], [], []
    for it in its:
        W, H = a.nmf.snapshots[it]
        src = alg.frames_to_samples(H, a.centers, a.ds.X.shape[1]) if a.single else H
        f, c = source_scores(a.ds, src)
        f1.append(f), corr.append(c)
        cos.append(float("nan") if a.single else mixing_cos(a.ds.A, W))
    return its, corr, cos, f1
