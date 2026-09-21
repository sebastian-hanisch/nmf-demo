"""Plotly-Visualisierungen der NMF-Demo: Elektrodenlayout, Spuren, die nicht-negative Matrix (gleichgerichtete Elektroden bzw. Spektrogramm), Fehlerverlauf und Zwischenstände der Faktorisierung, Mischspalten,
Aktivitäten gegen die wahren Neuronen, Raster, Verwechslungsmatrix, Kennzahlen-Balken, Sweeps, Fehlerkurve über k, Initialisierung und Szenen-Vergleich. Alle Figuren laufen durch `lock_axes`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nm_constants as C
from nm_scenario import electrode_positions

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f"
NEURON_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b")
METHOD_COLORS = {"nmf": TEAL, "ica": ORANGE, "sobi": PURPLE, "sca": GREEN, "pca": "#6d4c41", "electrode": GRAY}
METHOD_NAMES = {"nmf": "NMF", "ica": "ICA", "sobi": "SOBI", "sca": "SCA", "pca": "PCA", "electrode": "beste Elektrode"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def source_color(i):
    return NEURON_COLORS[i % len(NEURON_COLORS)]


def source_labels(ds):
    return [f"Neuron {i + 1}" for i in range(ds.n_neurons)]


def _window(t0, width):
    fs = C.SAMPLE_RATE
    return int(t0 * fs / 1000), int((t0 + width) * fs / 1000)


def build_layout(ds):
    """Elektroden (Quadrate auf y = 0) und Neuronen (Kreise, Größe = Spitzenamplitude)."""
    pos = electrode_positions(ds.n_electrodes)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j + 1}" for j in range(ds.n_electrodes)], textposition="bottom center", marker=dict(symbol="square", size=14, color=GRAY),
                             hoverinfo="skip"))
    for i in range(ds.n_neurons):
        x, y = C.NEURON_POSITIONS[i]
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[f"N{i + 1}"], textposition="top center", hoverinfo="skip", marker=dict(size=10 + 14 * C.NEURON_AMPLITUDES[i], color=source_color(i), opacity=0.85)))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.1, 1.1], title="Ort (willkürliche Einheit)", zeroline=False), yaxis=dict(range=[-0.15, 0.7], title="Abstand", zeroline=False),
                      showlegend=False)
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, colors=None, spike_times=None, height=None, normalise=True):
    """Gestapelte Spuren eines Zeitfensters [t0, t0 + width) in ms; optional Markierungen der Spitzen je Zeile."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    fig = go.Figure()
    n = len(arrays)
    scale_all = max(float(np.abs(a).max()) for a in arrays) if not normalise else None
    for r, (label, y) in enumerate(zip(labels, arrays)):
        scale = float(np.abs(y).max()) if normalise else scale_all
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + y[lo:hi] / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if spike_times is not None and r < len(spike_times):
            marks = [t for t in spike_times[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=color), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(180, 42 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title="Zeit [ms]"),
                      yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_spectrogram(V, centers, t0, width, title="Betragsspektrogramm"):
    """Betragsspektrogramm einer Elektrode im Zeitfenster (Frequenz nach oben, Helligkeit = Betrag)."""
    fs = C.SAMPLE_RATE
    t = centers * 1000.0 / fs
    sel = (t >= t0) & (t < t0 + width)
    freqs = np.arange(V.shape[0]) * fs / C.SPECTROGRAM_WINDOW / 1000.0
    fig = go.Figure(go.Heatmap(z=V[:, sel], x=t[sel], y=freqs, colorscale="Viridis", showscale=False, hoverinfo="skip"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]"), yaxis=dict(title="Frequenz [kHz]"))
    return lock_axes(fig)


def build_error_history(errors, snapshot_iterations=()):
    """Relativer Fehler ‖V − WH‖/‖V‖ je Iteration (logarithmisch); Punkte = die Zwischenstände, die in der Schrittansicht verglichen werden."""
    fig = go.Figure(go.Scatter(x=list(range(len(errors))), y=errors, mode="lines", line=dict(color=TEAL, width=2.5), hoverinfo="skip"))
    snaps = [i for i in snapshot_iterations if i < len(errors)]
    if snaps:
        fig.add_trace(go.Scatter(x=snaps, y=[errors[i] for i in snaps], mode="markers", marker=dict(size=9, color=RED), hoverinfo="skip"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title="Iteration"), yaxis=dict(title="relativer Fehler", type="log"))
    return lock_axes(fig)


def build_progress(iterations, corr, cos, f1):
    """Wie sich die Zerlegung über die Iterationen der Wahrheit nähert: Korrelation der Aktivitäten mit den wahren Neuronen, Kosinus der Mischspalten, Spitzen-F1 (an den Zwischenständen)."""
    fig = go.Figure()
    for y, name, color in ((corr, "Korrelation der Aktivitäten", TEAL), (cos, "Mischspalten (Kosinus)", ORANGE), (f1, "Spitzen-F1", PURPLE)):
        fig.add_trace(go.Scatter(x=[str(i) for i in iterations], y=y, mode="lines+markers", name=name, line=dict(color=color, width=2), hoverinfo="skip"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Iteration (0 = Start)", type="category"), yaxis=dict(range=[0, 1.05]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_mixing(A, W, idx, labels=None, x_labels=None):
    """Mischspalten je Neuron (Spalten, auf Summe 1 normiert): grau = wahre Spalte, farbig = die zugeordnete NMF-Komponente. `idx` = Komponente je Neuron (-1 = keine)."""
    m = A.shape[1]
    labels = labels or [f"Neuron {i + 1}" for i in range(m)]
    x_labels = x_labels or [f"E{j + 1}" for j in range(A.shape[0])]
    fig = make_subplots(rows=1, cols=m, subplot_titles=labels, shared_yaxes=True, horizontal_spacing=0.03)
    for i in range(m):
        col = A[:, i] / max(A[:, i].sum(), 1e-12)
        fig.add_trace(go.Bar(x=x_labels, y=col, marker_color=GRAY, hoverinfo="skip", showlegend=False), row=1, col=i + 1)
        if idx[i] >= 0:
            fig.add_trace(go.Bar(x=x_labels, y=W[:, idx[i]], marker_color=source_color(i), opacity=0.8, hoverinfo="skip", showlegend=False), row=1, col=i + 1)
    fig.update_layout(height=260, barmode="group", margin=dict(l=10, r=10, t=30, b=10))
    return lock_axes(fig)


def build_spectra(W, idx, n_neurons):
    """Einkanal: die Spektren der Komponenten (W, Summe 1) über der Frequenz, Farbe = zugeordnetes Neuron (grau = keinem zugeordnet)."""
    freqs = np.arange(W.shape[0]) * C.SAMPLE_RATE / C.SPECTROGRAM_WINDOW / 1000.0
    fig = go.Figure()
    owner = {c: i for i, c in enumerate(idx) if c >= 0}
    for c in range(W.shape[1]):
        i = owner.get(c, -1)
        fig.add_trace(go.Scatter(x=freqs, y=W[:, c], mode="lines", line=dict(color=source_color(i) if i >= 0 else GRAY, width=3), name=f"Komponente {c + 1}", hoverinfo="skip", showlegend=False))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Frequenz [kHz]"), yaxis=dict(title="Anteil"))
    return lock_axes(fig)


def build_activity(S, aligned, t0, width, labels=None):
    """Je Neuron eine Zeile: die wahre Aktivität (negative Spitze, grau) und die Aktivität der zugeordneten Komponente (farbig), beides auf die Spitzenhöhe normiert."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    m = len(S)
    labels = labels or [f"Neuron {i + 1}" for i in range(m)]
    fig = go.Figure()
    for r in range(m):
        offset = (m - 1 - r) * 1.3
        s = np.maximum(-S[r], 0)
        a = np.maximum(-aligned[r], 0)
        x = np.arange(lo, hi) * 1000.0 / fs
        fig.add_trace(go.Scatter(x=x, y=offset + s[lo:hi] / max(s.max(), 1e-12), mode="lines", line=dict(color=GRAY, width=3), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=offset + a[lo:hi] / max(a.max(), 1e-12), mode="lines", line=dict(color=source_color(r), width=1.4), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=max(200, 46 * m + 60), margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]"),
                      yaxis=dict(tickmode="array", tickvals=[(m - 1 - r) * 1.3 for r in range(m)], ticktext=labels, zeroline=False))
    return lock_axes(fig)


def build_raster(truth, detected, t0, width):
    """Spikes im Zeitfenster: je Neuron eine Zeile mit den wahren Spikes (oben, Strich) und den auf seiner Komponente erkannten (unten, Punkt)."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    m = len(truth)
    fig = go.Figure()
    for i in range(m):
        base = (m - 1 - i) * 1.0
        tsel = np.array([t for t in truth[i] if lo <= t < hi])
        dsel = np.array([t for t in detected[i] if lo <= t < hi])
        if len(tsel):
            fig.add_trace(go.Scatter(x=tsel * 1000.0 / fs, y=np.full(len(tsel), base + 0.25), mode="markers", marker=dict(symbol="line-ns-open", size=14, color=source_color(i), line=dict(width=2)), hoverinfo="skip", showlegend=False))
        if len(dsel):
            fig.add_trace(go.Scatter(x=dsel * 1000.0 / fs, y=np.full(len(dsel), base - 0.05), mode="markers", marker=dict(symbol="circle", size=8, color=source_color(i)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=60 + 60 * m, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]", range=[lo * 1000.0 / fs, hi * 1000.0 / fs]),
                      yaxis=dict(tickmode="array", tickvals=[(m - 1 - i) * 1.0 + 0.1 for i in range(m)], ticktext=[f"Neuron {i + 1}" for i in range(m)], zeroline=False))
    return lock_axes(fig)


def build_method_bars(scores, baseline=None):
    """Spitzen-F1 aller Verfahren (und, gestrichelt, der Schwelle als Referenz)."""
    names = [n for n in ("nmf", "ica", "sobi", "sca", "pca", "electrode") if n in scores]
    f1 = [scores[n]["f1"] for n in names]
    fig = go.Figure(go.Bar(x=[METHOD_NAMES[n] for n in names], y=f1, marker_color=[METHOD_COLORS[n] for n in names], text=[f"{v:.2f}" for v in f1], textposition="outside", hoverinfo="skip"))
    if baseline is not None:
        fig.add_hline(y=baseline, line=dict(color=RED, dash="dash"), annotation_text=f"Schwelle {baseline:.2f}", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.15]), showlegend=False)
    return lock_axes(fig)


def build_confusion(corr_matrix, idx):
    """Korrelation |r| (Zeilen = wahre Neuronen, Spalten = Komponenten, nach der optimalen Zuordnung sortiert, nicht zugeordnete hinten); die Diagonale ist die richtige Trennung."""
    m, k = corr_matrix.shape
    order = [c for c in idx if c >= 0] + [c for c in range(k) if c not in idx]
    Z = corr_matrix[:, order]
    fig = go.Figure(go.Heatmap(z=Z, x=[f"Komp. {c + 1}" for c in order], y=[f"Neuron {i + 1}" for i in range(m)], colorscale="Blues", zmin=0, zmax=1, text=np.round(Z, 2), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 42 * m, margin=dict(l=10, r=10, t=10, b=10))
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None, with_comparators=True):
    """Links: Spitzen-F1 von NMF (mit Streuung) und - bei Reglern der Daten - ICA, SOBI, SCA; rechts: Korrelation und Mischspalten-Kosinus der NMF sowie ihr Fehler gegen den Fehler mit den wahren Mischspalten."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spitzen-F1", "NMF im Detail"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    y, sd = np.array([r["nmf"] for r in rows]), np.array([r["nmf_std"] for r in rows])
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=list(y + sd) + list(y - sd)[::-1], fill="toself", fillcolor=TEAL, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=y, mode="lines+markers", name="NMF", line=dict(color=TEAL, width=3), hoverinfo="skip"), row=1, col=1)
    if with_comparators:
        for name in ("ica", "sobi", "sca"):
            fig.add_trace(go.Scatter(x=xs, y=[r[name] for r in rows], mode="lines+markers", name=METHOD_NAMES[name], line=dict(color=METHOD_COLORS[name], width=1.8), hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=[r["nmf_corr"] for r in rows], mode="lines+markers", name="Korrelation (NMF)", line=dict(color=BLUE, width=2), hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=[r["nmf_angle"] for r in rows], mode="lines+markers", name="Mischspalten-Kosinus", line=dict(color=ORANGE, width=2), hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_error_gap(rows, xlabel):
    """Relativer Fehler der NMF (klein = passt gut) gegen den Fehler mit den wahren Mischspalten: liegt die NMF darunter, hat sie eine besser passende - aber falsche - Zerlegung gefunden."""
    xs = [r["x"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["nmf_error"] for r in rows], mode="lines+markers", name="NMF", line=dict(color=TEAL, width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["error_true"] for r in rows], mode="lines+markers", name="mit den wahren Mischspalten", line=dict(color=GRAY, width=2, dash="dash"), hoverinfo="skip"))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=xlabel), yaxis=dict(title="relativer Fehler"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_rank(rows):
    """Mittlere Fehlerkurve über die Komponentenzahl k je wahrer Neuronenzahl (senkrechte Striche: wahre Zahl)."""
    fig = go.Figure()
    for r in rows:
        color = source_color(r["m"] - 2)
        fig.add_trace(go.Scatter(x=list(range(1, len(r["curve"]) + 1)), y=r["curve"], mode="lines+markers", name=f"{r['m']} Neuronen", line=dict(color=color, width=2), hoverinfo="skip"))
        fig.add_vline(x=r["m"], line=dict(color=color, dash="dot", width=1))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Komponentenzahl k", dtick=1), yaxis=dict(title="relativer Fehler", type="log"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_init(per_seed):
    """Zufällige Starts (Punkte, je Aufnahme eine Farbe): Fehler gegen Spitzen-F1; Rauten = NNDSVD. Der Start mit dem kleinsten Fehler ist nicht immer der mit dem besten F1."""
    fig = go.Figure()
    for i, p in enumerate(per_seed):
        runs = p["runs"]
        fig.add_trace(go.Scatter(x=[r[0] for r in runs], y=[r[1] for r in runs], mode="markers", marker=dict(size=8, color=source_color(i), opacity=0.8), name=f"Aufnahme {i + 1}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[p["err_nndsvd"]], y=[p["nndsvd"]], mode="markers", marker=dict(size=13, symbol="diamond", color=source_color(i), line=dict(color="black", width=1)), showlegend=False, hoverinfo="skip"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="relativer Fehler des Starts"), yaxis=dict(title="Spitzen-F1", range=[0, 1.05]), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_scenes(rows):
    """NMF, ICA, SOBI und SCA: Spitzen-F1 je Szene; Balken = Mittel, Fehlerbalken = Spanne über die Sweep-Datensätze."""
    labels = [r["scene"].replace(" (", "<br>(").replace(": ", ":<br>") for r in rows]
    fig = go.Figure()
    for name in ("nmf", "ica", "sobi", "sca"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", textfont=dict(size=9), hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[name + "_max"] - r[name] for r in rows], arrayminus=[r[name] - r[name + "_min"] for r in rows])))
    fig.update_layout(height=520, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Neuronen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.6))
    return lock_axes(fig)
