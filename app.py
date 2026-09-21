"""Nicht-negative Matrixfaktorisierung (NMF) - Nicht-Negativität statt Unabhängigkeit - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die NMF - und lässt stattdessen das Beispiel wachsen.
Siebtes und letztes Stück der Quellentrennung-Linie der "Konzepte"-Reihe: NMF lockert die ICA-Annahme "Unabhängigkeit" zu "Nicht-Negativität", ist einkanalfähig - und trennt hier oft schlechter als ICA, SOBI und SCA.
Was sie trotzdem kann und wo sie scheitert, wird hier gemessen. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import nm_algorithm as alg
import nm_constants as C
from nm_evaluation import SWEEP_LABELS, Settings, analyse_for, component_view, init_table, make_dataset, progress, rank_table, scene_table, sweep, verdict
from nm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from nm_visualization import (
    build_activity,
    build_confusion,
    build_error_gap,
    build_error_history,
    build_init,
    build_layout,
    build_method_bars,
    build_mixing,
    build_progress,
    build_rank,
    build_raster,
    build_scenes,
    build_spectra,
    build_spectrogram,
    build_sweep,
    build_traces,
    source_color,
    source_labels,
)

st.set_page_config(page_title="NMF – Sebastian Hanisch", layout="wide")

WINDOW_WIDTH_MS = 60
SWEEP_OPTIONS = {"n": "Anzahl Elektroden", "m": "Anzahl Neuronen", "noise": "Rauschen", "rate_scale": "Feuerrate", "similarity": "Wellenform-Ähnlichkeit", "jitter": "Amplitudenschwankung",
                 "synchrony": "Synchronität", "threshold": "Rauschschwelle der Gleichrichtung", "sparsity": "Sparsität λ", "max_iter": "Iterationen"}


def _pct(x):
    return "–" if np.isnan(x) else f"{x:.0%}"


@st.cache_data(show_spinner=False)
def _dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony, seed):
    return make_dataset(m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony, seed)


@st.cache_data(show_spinner=False)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _progress(data_params, settings):
    return progress(analyse_for(data_params, settings, comparators=False))


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony = base
    return sweep(parameter, settings=settings, m=m, n=n, rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples, synchrony=synchrony)


@st.cache_data(show_spinner=False)
def _rank(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony = base
    return rank_table(settings, n=max(n, 2), rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples, synchrony=synchrony)


@st.cache_data(show_spinner=False)
def _init(base, settings):
    m, n, rate_scale, similarity, jitter, noise, n_samples, synchrony = base
    return init_table(settings, m=m, n=n, rate_scale=rate_scale, similarity=similarity, jitter=jitter, noise=noise, n_samples=n_samples, synchrony=synchrony)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    return scene_table(settings, noise=base[5], n_samples=base[6])


st.title("🧱 NMF – Nicht-Negativität statt Unabhängigkeit")
st.markdown(
    """
Die Quellentrennung-Linie begann mit der **ICA**: aus den Elektrodensignalen die Neuronen herausrechnen, weil sie **statistisch unabhängig** und nicht Gauß'sch sind. Die **nicht-negative Matrixfaktorisierung (NMF)** setzt woanders an:
eine Matrix $V \\ge 0$ wird als Produkt $V \\approx W H$ zweier **nicht-negativer** Matrizen geschrieben - Spalten von $W$ sind die Muster (hier: Mischspalten oder Spektren), Zeilen von $H$ ihre Aktivitäten. Weil nichts abgezogen werden darf,
entstehen "Teile" statt Differenzen; und weil sie keine Unabhängigkeit verlangt, funktioniert sie auch mit **abhängigen Quellen** und - über das Spektrogramm - mit **einer einzigen Elektrode**.
Elektrodensignale sind aber bipolar (negative Spitze, positiver Nachschlag): sie müssen erst **gleichgerichtet** werden. Ob das gutgeht, zeigt die Demo - mit Siegen und Niederlagen. Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - siebtes und letztes Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Array ist dasselbe wie in ICA-, SOBI-, SCA-, Pipeline- und Abgleich-Demo; neu sind die Synchronität der Neuronen (abhängige Quellen) und Ähnlichkeit bis 2 (sehr verschieden breite Spitzen)."
)

with st.expander("So funktioniert die NMF", expanded=True):
    st.markdown(
        """
1. **Nicht-negativ machen.** Mehrere Elektroden: je Elektrode $V_{jt} = \\max(-(x_{jt} - \\tilde x_j) - c\\,\\sigma_j,\\, 0)$ - nur die **negative Spitze** zählt (oder der Betrag beider Phasen), abzüglich einer Rauschschwelle $c\\,\\sigma_j$
   (ohne Schwelle wird auch das Rauschen gleichgerichtet und bildet eine positive Grundlinie). Eine Elektrode: das **Betragsspektrogramm** (Kurzzeit-Fouriertransformation). Dann gilt näherungsweise $V \\approx W H$ mit $W \\ge 0$ (Mischspalten bzw. Spektren) und $H \\ge 0$ (Aktivitäten).
2. **Faktorisieren.** Multiplikative Updates (Lee und Seung): $H \\leftarrow H \\odot (W^\\top V) \\oslash (W^\\top W H)$, $W \\leftarrow W \\odot (V H^\\top) \\oslash (W H H^\\top)$ - der Fehler $\\lVert V - WH \\rVert$ sinkt in jeder Iteration, Nicht-Negativität bleibt erhalten.
   Start: zufällig (beste von fünf Neustarts) oder **NNDSVD** (aus den Singulärvektoren, deterministisch). Optional eine **Sparsitäts-Strafe** $\\lambda$ auf $H$.
3. **Komponenten lesen.** Jede Zeile von $H$ ist die Aktivität einer Komponente; ihre Spitzen sind die Spikes des zugeordneten Neurons (Zuordnung per Korrelation, wie bei ICA, SOBI und SCA - für alle Verfahren dieselbe Bewertung).
4. **Komponentenzahl.** Bekannt (= Neuronenzahl) oder unbekannt: der Fehler über $k$ hat einen **Knick**, wo weitere Komponenten kaum noch etwas erklären.

Was NMF **verlangt**: nicht-negative Daten und dass sich die Muster additiv überlagern. Was sie **nicht** verlangt: Unabhängigkeit, mehr Elektroden als Neuronen (nur Rang) - und ein Kanal genügt. Was sie **nicht garantiert**: eine **eindeutige** Zerlegung - es gibt oft
mehrere nicht-negative Faktorisierungen mit ähnlichem Fehler, und die mit dem kleinsten Fehler muss nicht die richtige sein.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_neurons = st.slider(
        "Neuronen", *bounds("n_neurons_slider"), key="n_neurons_slider",
        help="Spitzenartige Quellen wie in den anderen Demos. Bei 4 Elektroden Spitzen-F1 der NMF (ICA): 2 Neuronen 0.87 (1.00), 3 Neuronen 0.84 (1.00), 4 Neuronen 0.77 (1.00), 5 Neuronen 0.59 (0.77); SCA bleibt bei 0.99-1.00. "
             "Mit jedem Neuron liegen die positiven Mischspalten dichter beieinander.",
    )
    n_electrodes = st.slider(
        "Elektroden", *bounds("n_electrodes_slider"), key="n_electrodes_slider",
        help="Bei einer Elektrode arbeitet die NMF auf dem Betragsspektrogramm (bei vier Neuronen 0.38; die Schwelle auf der Elektrode 0.40; ICA 0.16, SOBI 0.16, SCA 0.05: sie brauchen mehrere Kanäle). Bei zwei Elektroden 0.64 (ICA 0.36, SOBI 0.35, SCA 0.97), "
             "bei drei 0.72 (0.69 / 0.69 / 0.99), ab vier 0.77 (ICA, SOBI, SCA 1.00).",
    )
    rate_scale = st.slider(
        "Feuerrate (Faktor)", *bounds("rate_slider"), key="rate_slider", step=0.25,
        help="Faktor auf die Feuerraten (20-35 Hz). Faktor 0.25 / 0.5 / 1 / 2: NMF 0.78 / 0.80 / 0.77 / 0.79 - die Feuerrate ändert kaum etwas; seltenere Spikes (weniger Überlappung) helfen der NMF nicht, weil ihre Grenze die dicht beieinander liegenden Mischspalten sind.",
    )
    similarity = st.slider(
        "Wellenform-Ähnlichkeit", *bounds("similarity_slider"), key="similarity_slider", step=0.05,
        help="0 = alle Neuronen dieselbe Form, 1 = die gewohnten Breiten, 2 = die Breitenunterschiede doppelt so groß. Mit mehreren Elektroden ohne Wirkung auf die NMF (0.77 bei 0, 0.77 bei 1, 0.76 bei 2); SOBI leidet bei 0 (0.58). "
             "Mit einer Elektrode entscheidet sie: zwei Neuronen 0.50 bei 1 (Schwelle 0.66), 0.97 bei 2.",
    )
    jitter = st.slider(
        "Amplitudenschwankung", *bounds("jitter_slider"), key="jitter_slider", step=0.05,
        help="Streuung der Spitzenhöhe von Spike zu Spike (relativ). Bei 0 / 0.1 / 0.2 / 0.3: NMF 0.77 / 0.80 / 0.81 / 0.81 (leicht besser: schwankende Höhen machen die Aktivitäten unterscheidbarer); ICA, SOBI und SCA bleiben bei 0.99-1.00.",
    )
    synchrony = st.slider(
        "Synchronität", *bounds("synchrony_slider"), key="synchrony_slider", step=0.05,
        help="Anteil der Spikes, den die Neuronen 2 und weitere mit Neuron 1 teilen (0 = unabhängig). ICA setzt Unabhängigkeit voraus: Bei 0 / 0.3 / 0.6 / 0.9 erreicht die NMF 0.77 / 0.87 / 0.91 / 0.98, ICA 1.00 / 1.00 / 0.94 / 0.66, "
             "SOBI 1.00 / 1.00 / 0.99 / 0.80, SCA 1.00 / 1.00 / 0.92 / 0.72. Bedenken: bei 0.9 erreicht auch die beste Einzelelektrode 0.92 - bei so viel Gleichzeitigkeit ist die Aufgabe leicht.",
    )
    noise = st.slider(
        "Rauschen", *bounds("noise_slider"), key="noise_slider", step=0.05,
        help="Sensorrauschen relativ zum Neuronen-Signal. Bei 0.05 / 0.2 / 0.5 / 1.0: NMF (Schwelle 2) 0.77 / 0.75 / 0.71 / 0.71; ICA 1.00 / 0.99 / 0.83 / 0.43, SOBI 1.00 / 0.79 / 0.53 / 0.38, SCA 1.00 / 1.00 / 0.94 / 0.86 - "
             "die NMF verliert am wenigsten von den dreien mit ICA und SOBI, SCA bleibt aber besser.",
    )
    n_samples = st.slider(
        "Länge der Aufnahme", *bounds("n_samples_slider"), key="n_samples_slider", step=1000,
        help="Abtastwerte bei 10 kHz. Mehr Länge = mehr Spikes, aber auch längere Rechenzeit (40000 statt 20000 Abtastwerte etwa das Vierfache für die NMF).",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**NMF**")
    if n_electrodes > 1:
        mode = st.selectbox(
            "Gleichrichtung", alg.RECTIFY_MODES, key="mode_select", format_func=lambda x: alg.RECTIFY_LABELS[x],
            help="Wie die bipolaren Signale nicht-negativ werden: nur die negative Spitze (Standard) oder der Betrag beider Phasen. Spitzen-F1 fast gleich (0.77 / 0.78), aber die Korrelation mit den wahren Neuronen fällt beim Betrag von 0.79 auf 0.55: "
                 "Spitze und Nachschlag werden zu einer Aktivität verschmolzen.",
        )
        threshold = st.slider(
            "Rauschschwelle c (Rausch-σ)", *bounds("threshold_slider"), key="threshold_slider", step=0.5,
            help="Vor dem Kappen abgezogen: ohne Schwelle wird das Rauschen mitgleichgerichtet und bildet eine positive Grundlinie. Bei Rauschen 0.05 kaum ein Unterschied (0.76 bei 0, 0.77 bei 2); bei Rauschen 1.0 entscheidet sie: 0.52 (0), 0.63 (1), 0.71 (2), 0.74 (3).",
        )
        st.session_state["_mode_kept"] = mode
        st.session_state["_threshold_kept"] = threshold
    else:
        mode = st.session_state.get("_mode_kept", C.DEFAULT_MODE)
        threshold = float(st.session_state.get("_threshold_kept", C.DEFAULT_THRESHOLD))
    k_mode = st.selectbox(
        "Komponentenzahl", C.K_MODES, key="k_mode_select", format_func=lambda x: C.K_MODE_LABELS[x],
        help="Bekannt: so viele Komponenten wie Neuronen. Unbekannt: der Fehler über k hat einen Knick. Bei 2, 3 und 4 Neuronen trifft er in allen fünf Test-Aufnahmen, bei 5 immer nur 4 (das fünfte Neuron senkt den Fehler zu wenig).",
    )
    init = st.selectbox(
        "Initialisierung", alg.INITS, key="init_select", format_func=lambda x: alg.INIT_LABELS[x],
        help="Zufällig: fünf Neustarts, der mit dem kleinsten Fehler zählt (Spitzen-F1 0.77); NNDSVD: deterministisch aus den Singulärvektoren (0.73). Die einzelnen Zufallsstarts streuen im F1 um etwa 0.03, "
             "und ein kleinerer Fehler geht meist mit einem besseren Ergebnis einher.",
    )
    sparsity = st.slider(
        "Sparsität λ", *bounds("sparsity_slider"), key="sparsity_slider", step=0.5,
        help="L1-Strafe auf die Aktivitäten H (0 = klassische NMF): bevorzugt Zerlegungen, in denen zu jedem Zeitpunkt nur wenige Komponenten aktiv sind (wie SCA). Bei 0 / 0.5 / 1 / 3: 0.77 / 0.81 / 0.80 / 0.79 und Mischspalten-Kosinus 0.95 / 0.98 / 0.98 / 0.98; "
             "der Fehler steigt dabei (0.013 → 0.086) - er liegt dann nicht mehr unter dem mit den wahren Mischspalten (0.049).",
    )
    iterations = st.slider(
        "Iterationen", *bounds("iterations_slider"), key="iterations_slider", step=10,
        help="Höchstzahl der multiplikativen Updates. Mehr Iterationen passen besser (Fehler 0.055 bei 20, 0.017 bei 100, 0.010 bei 1000), trennen aber schlechter: Spitzen-F1 0.75 / 0.80 / 0.77 / 0.74 bei 20 / 100 / 300 / 1000.",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Spikezeiten, Rauschen und Zufallsstart.")

sync_query_params({
    "n_neurons_slider": int(n_neurons), "n_electrodes_slider": int(n_electrodes), "rate_slider": float(rate_scale), "similarity_slider": float(similarity), "jitter_slider": float(jitter),
    "synchrony_slider": float(synchrony), "noise_slider": float(noise), "n_samples_slider": int(n_samples), "mode_select": mode, "threshold_slider": float(threshold), "k_mode_select": k_mode, "init_select": init,
    "sparsity_slider": float(sparsity), "iterations_slider": int(iterations), "seed_input": int(seed),
})

data_params = (int(n_neurons), int(n_electrodes), float(rate_scale), float(round(similarity, 2)), float(round(jitter, 2)), float(noise), int(n_samples), float(round(synchrony, 2)), int(seed))
settings = Settings(mode=mode, threshold=float(threshold), k_mode=k_mode, init=init, sparsity=float(sparsity), max_iter=int(iterations))
with st.spinner("Faktorisiere..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
level, code, vd = verdict(analysis)
m_n, n_el = ds.n_neurons, ds.n_electrodes
single = analysis.single
labels = source_labels(ds)
colors = [source_color(i) for i in range(m_n)]
res = analysis.nmf
idx, corr_matrix, aligned, detected = component_view(analysis)
data_key = data_params + (settings,)

# --- NMF in Aktion ---------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 NMF in Aktion")
STEP_LABELS = {1: "1 · Signal", 2: "2 · Spektrogramm" if single else "2 · Gleichrichten", 3: "3 · Faktorisieren", 4: "4 · Komponenten", 5: "5 · Ergebnis"}
if "nm_step" not in st.session_state or st.session_state.get("nm_step_owner") != data_key:
    st.session_state["nm_step"] = 1
    st.session_state["nm_step_owner"] = data_key
duration_ms = ds.X.shape[1] * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="nm_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Zeitreihen zeigen.")
view_slot = st.empty()


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Die Elektrodensignale** (▼ = wahre Spitzen der Neuronen)")
            c1.plotly_chart(build_traces([f"E{j + 1}" for j in range(n_el)], list(ds.X), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_electrodes")
            c2.markdown("**Ort von Neuronen und Elektroden**")
            c2.plotly_chart(build_layout(ds), width="stretch", key="step_layout")
            st.markdown("**Die wahren Neuronen** (unbekannt in der Praxis)")
            st.plotly_chart(build_traces(labels, list(ds.S), window, WINDOW_WIDTH_MS, colors, [ds.spike_times[i] for i in range(m_n)]), width="stretch", key="step_sources")
        elif current_step == 2:
            if single:
                st.markdown("**Betragsspektrogramm der einen Elektrode** (Kurzzeit-Fouriertransformation, Fenster 3.2 ms; Frequenz nach oben)")
                st.plotly_chart(build_spectrogram(analysis.V, analysis.centers, window, WINDOW_WIDTH_MS), width="stretch", key="step_spectrogram")
            else:
                st.markdown(f"**Gleichgerichtete Elektrodensignale V = max(−x − {threshold:g}·σ, 0)** ({alg.RECTIFY_LABELS[mode]}; nur noch Werte ≥ 0)" if mode == "negative" else
                            f"**Gleichgerichtete Elektrodensignale V = max(|x| − {threshold:g}·σ, 0)** (Betrag beider Phasen)")
                st.plotly_chart(build_traces([f"E{j + 1}" for j in range(n_el)], list(analysis.V), window, WINDOW_WIDTH_MS, normalise=False), width="stretch", key="step_rectified")
        elif current_step == 3:
            c1, c2 = st.columns(2)
            c1.markdown("**Fehler ‖V − WH‖/‖V‖ je Iteration** (Punkte: die verglichenen Zwischenstände)")
            c1.plotly_chart(build_error_history(res.errors, sorted(res.snapshots)), width="stretch", key="step_error")
            its, pc, pcos, pf1 = _progress(data_params, settings)
            c2.markdown("**Nähe zur Wahrheit an den Zwischenständen**")
            c2.plotly_chart(build_progress(its, pc, pcos, pf1), width="stretch", key="step_progress")
        elif current_step == 4:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Aktivität je Neuron: wahr (grau) gegen die zugeordnete Komponente (farbig)**")
            c1.plotly_chart(build_activity(ds.S, aligned, window, WINDOW_WIDTH_MS), width="stretch", key="step_activity")
            if single:
                c2.markdown("**Spektren der Komponenten** (farbig = zugeordnetes Neuron, grau = keinem zugeordnet)")
                c2.plotly_chart(build_spectra(res.W, idx, m_n), width="stretch", key="step_spectra")
            else:
                c2.markdown("**Mischspalten: wahr (grau) und gefunden (farbig)**")
                c2.plotly_chart(build_mixing(ds.A, res.W, idx), width="stretch", key="step_mixing")
        else:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Raster: wahre Spikes (Striche) und die auf den Komponenten erkannten (Punkte)**")
            c1.plotly_chart(build_raster([ds.spike_times[i] for i in range(m_n)], detected, window, WINDOW_WIDTH_MS), width="stretch", key="step_raster")
            c2.markdown("**Korrelation** (Zeilen: wahre Neuronen, Spalten: Komponenten)")
            c2.plotly_chart(build_confusion(corr_matrix, idx), width="stretch", key="step_confusion")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

nmf_s = analysis.scores["nmf"]
if step == 1:
    st.caption(f"{m_n} Neuronen feuern; {n_el} Elektrode(n) messen Mischungen. Synchronität {synchrony:.0%}: " + ("die Neuronen feuern unabhängig." if synchrony == 0 else "ein Teil der Spikes fällt mit denen von Neuron 1 zusammen (die Quellen sind abhängig).")
               + " Die NMF sieht nur die Elektrodensignale.")
elif step == 2:
    st.caption(("Ein Kanal: die Faktorisierung zerlegt das Zeit-Frequenz-Bild in Spektren × Aktivitäten. Neuronen mit verschieden breiten Spitzen haben verschiedene Spektren (schmale Spitze = breites Spektrum)." if single else
                f"Nur die negative Phase bleibt (Schwelle {threshold:g} σ), der Nachschlag fällt weg. Damit ist V ≈ Mischspalten × Aktivität - aber nur näherungsweise: Überlappungen addieren sich nur in der negativen Phase, Rauschen wird mitgleichgerichtet."
                if mode == "negative" else "Beide Phasen bleiben (Betrag): Spitze und Nachschlag zählen als Aktivität; das Rauschen ebenso, soweit es über der Schwelle liegt."))
elif step == 3:
    st.caption(f"{res.n_iter} Iterationen ({'konvergiert' if res.converged else 'Höchstzahl erreicht'}), Fehler {res.error:.3f}. " + (f"Mit den wahren Mischspalten wäre der Fehler {analysis.error_true:.3f}: " + ("die NMF hat eine besser passende Zerlegung gefunden als die wahre - der Fehler allein zeigt nicht, ob sie richtig ist. " if res.error < analysis.error_true else "die NMF passt schlechter als die wahre Zerlegung. ") if not single else "")
               + "Rechts: Korrelation, Mischspalten und Spitzen-F1 an den Zwischenständen - sie steigen meist schnell und können bei weiteren Iterationen wieder fallen, obwohl der Fehler weiter sinkt.")
elif step == 4:
    st.caption(f"Komponentenzahl {analysis.k}. Korrelation der Aktivitäten mit den wahren Neuronen {nmf_s['corr']:.2f}" + ("" if single else f", Mischspalten-Kosinus {nmf_s['angle']:.2f} (größte Ähnlichkeit zweier wahrer Mischspalten: {analysis.column_similarity:.2f} - je näher an 1, desto schwerer sind sie zu trennen)") + ".")
else:
    st.caption(f"Spitzen-F1 {nmf_s['f1']:.2f}: je Neuron die Spitzen seiner Komponente gegen seine wahren Spitzen (Toleranz ±4 Abtastwerte), gemittelt. Striche ohne Punkt darunter sind verpasste Spikes; Punkte ohne Strich darüber sind falsch zugeordnet oder erfunden.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die NMF gefunden hat - im Vergleich mit ICA, SOBI und SCA")
st.caption(
    "Spitzen-F1: je Neuron die Spitzen seiner geschätzten Aktivität (bei ICA, SOBI, SCA seiner Schätzspur) gegen die wahren Spitzenzeiten, gemittelt über die Neuronen - für alle Verfahren dieselbe Definition; "
    "Zuordnung der Komponenten zu den Neuronen per Korrelation. Als Referenzen: PCA, die beste Einzelelektrode und die Schwelle (alle erkannten Spitzen jedem Neuron zugerechnet)."
)
best_other = vd["best"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spitzen-F1 (NMF)", f"{nmf_s['f1']:.2f}", delta=f"{nmf_s['f1'] - best_other:+.2f} ggü. bestem von ICA/SOBI/SCA" if not np.isnan(best_other) else None, delta_color="normal",
          help="Mittlerer Spitzen-F1 der Neuronen; im Delta der Abstand zum besten der drei Vergleichsverfahren.")
m2.metric("Korrelation mit den Neuronen", f"{nmf_s['corr']:.2f}", delta=f"Einzelelektrode {analysis.scores['electrode']['corr']:.2f}", delta_color="off", help="Mittlere |Korrelation| der zugeordneten Komponenten mit den wahren Neuronen; darunter die der besten Einzelelektrode.")
m3.metric("Mischspalten-Kosinus", "–" if single else f"{nmf_s['angle']:.2f}", delta="eine Elektrode: keine Mischspalten" if single else f"größte Spaltenähnlichkeit {analysis.column_similarity:.2f}", delta_color="off",
          help="Mittlere Kosinus-Ähnlichkeit der gefundenen mit den wahren Mischspalten (1 = richtig); darunter die größte Ähnlichkeit zweier wahrer Spalten.")
m4.metric("Fehler der Zerlegung", f"{res.error:.3f}", delta="Spektrogramm" if single else f"mit wahren Mischspalten {analysis.error_true:.3f}", delta_color="off",
          help="Relativer Fehler ‖V − WH‖/‖V‖ der NMF; darunter der Fehler, den die Zerlegung mit den wahren Mischspalten erreichen würde (ein kleinerer NMF-Fehler heißt: besser passend, nicht: richtig).")

_t = vd
_others = f"ICA {_t.get('ica_f1', float('nan')):.2f}, SOBI {_t.get('sobi_f1', float('nan')):.2f}, SCA {_t.get('sca_f1', float('nan')):.2f}"
if code == "nmf_wins":
    st.success(f"✅ Die NMF trennt hier am besten: Spitzen-F1 {_t['f1']:.2f} gegen {_others}. Sie braucht keine Unabhängigkeit - das ist der Fall, in dem ICA und die anderen leiden. "
               f"Bedenken: die beste Einzelelektrode erreicht {_t['electrode_f1']:.2f}; bei stark gleichzeitigem Feuern ist die Aufgabe ohnehin leicht.")
elif code == "comparable":
    st.success(f"✅ Gleichauf: Spitzen-F1 der NMF {_t['f1']:.2f} gegen {_others}.")
elif code == "others_win":
    st.warning(f"⚠️ Ein anderes Verfahren trennt besser: Spitzen-F1 der NMF {_t['f1']:.2f} gegen {_others}. Die NMF hat die gleichgerichteten Daten mit Fehler {_t['error']:.3f} erklärt"
               + (f" - die Zerlegung mit den wahren Mischspalten wäre schlechter angepasst ({_t['error_true']:.3f}): sie findet eine besser passende, aber falsche Faktorisierung. " if _t['error'] < _t['error_true'] else ". ")
               + f"Die Mischspalten der Neuronen liegen dicht beieinander (größte Ähnlichkeit zweier Spalten {_t['column_similarity']:.2f}); Nicht-Negativität allein trennt sie nicht eindeutig.")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen kostet: Spitzen-F1 {_t['f1']:.2f} gegen {_t['f1'] + _t['drop']:.2f} bei geringem Rauschen; {_others}. Die Rauschschwelle der Gleichrichtung ({threshold:g} σ) begrenzt die Grundlinie - ohne sie wäre es noch schlechter.")
elif code == "too_few_electrodes":
    st.warning(f"⚠️ Weniger Elektroden ({_t['n']}) als Neuronen ({_t['m']}): die Zerlegung hat höchstens Rang {_t['n']}, mehr Komponenten sind nicht eindeutig. Spitzen-F1 {_t['f1']:.2f} gegen {_others} - SCA ist genau für diesen Fall gebaut.")
elif code == "wrong_k":
    st.warning(f"⚠️ Der Knick der Fehlerkurve liegt bei {_t['k']} Komponenten, wahr sind {_t['m']}: das kleinste Neuron senkt den Fehler zu wenig, um aufzufallen. Spitzen-F1 {_t['f1']:.2f}; {_others}.")
elif code == "single_works":
    st.success(f"✅ Mit einer einzigen Elektrode trennt die NMF: Spitzen-F1 {_t['f1']:.2f} gegen {_t['baseline']:.2f} für die bloße Schwelle (alle Spitzen jedem Neuron zugerechnet). ICA, SOBI und SCA brauchen mehrere Kanäle ({_others}). "
               "Die Neuronen haben hier deutlich verschieden breite Spitzen und damit verschiedene Spektren.")
elif code == "single_weak":
    st.warning(f"⚠️ Mit einer Elektrode und ähnlich breiten Spitzen unterscheiden sich die Spektren kaum: Spitzen-F1 {_t['f1']:.2f}, nicht besser als die bloße Schwelle ({_t['baseline']:.2f}). "
               "Die NMF findet die Spikes, ordnet sie den Neuronen aber kaum zu - mehr Breitenunterschied (Ähnlichkeit bis 2) würde helfen.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Raster: wahre Spikes (Striche) und die auf den NMF-Komponenten erkannten (Punkte)**")
    st.plotly_chart(build_raster([ds.spike_times[i] for i in range(m_n)], detected, window, WINDOW_WIDTH_MS), width="stretch", key="res_raster")
with t2:
    st.markdown("**Spitzen-F1: NMF, ICA, SOBI, SCA und die Referenzen**")
    st.plotly_chart(build_method_bars(analysis.scores, analysis.baseline_f1 if single else None), width="stretch", key="res_bars")
st.caption("Punkte in der Zeile eines Neurons, ohne dass darüber ein Strich steht, sind falsch zugeordnet. ICA, SOBI und SCA sehen bei einer Elektrode nur eine Quelle (ICA, SOBI) bzw. haben keine Richtung zu schätzen (SCA) - ihr F1 dort ist der Zufall der Zuordnung. "
           f"Rechenzeit der NMF für diese Aufnahme: {analysis.seconds['nmf']:.2f} s ({len(analysis.start_errors)} Start(s)).")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Elektroden, Neuronen, Synchronität, Rauschen und den NMF-Reglern ab?")
sweep_options = [p for p in SWEEP_OPTIONS if p != "threshold" or n_el > 1]
if st.session_state.get("sweep_select") not in sweep_options:
    st.session_state["sweep_select"] = sweep_options[0]
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", sweep_options, format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"n": int(n_electrodes), "m": int(n_neurons), "noise": float(noise), "rate_scale": float(rate_scale), "similarity": float(similarity), "jitter": float(jitter), "synchrony": float(synchrony),
           "threshold": float(threshold), "sparsity": float(sparsity), "max_iter": int(iterations)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert bis etwa eine Minute)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows = _sweep(sweep_param, data_params[:8], settings)
    with_cmp = sweep_param not in ("threshold", "sparsity", "max_iter")
    st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current, with_comparators=with_cmp), width="stretch", key="sweep_chart")
    st.markdown("**Fehler der NMF gegen den Fehler mit den wahren Mischspalten** (liegt die NMF darunter, hat sie eine besser passende, aber falsche Zerlegung gefunden)")
    st.plotly_chart(build_error_gap(rows, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_error")
    st.caption("Mittel und Streuung (Band) über 5 feste Sweep-Datensätze (getrennt vom Seed oben); alle anderen Regler wie in der Seitenleiste. Die Regler der NMF (Schwelle, Sparsität, Iterationen) betreffen nur sie (keine Vergleichslinien). "
               "Bei einer Elektrode gibt es keine Mischspalten und keinen Vergleichsfehler.")

st.markdown("---")

# --- Komponentenzahl ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Komponentenzahl unbekannt: der Knick der Fehlerkurve")
if st.button("Fehlerkurven für 2-5 Neuronen berechnen (dauert etwa eine halbe Minute)", key="rank_start"):
    st.session_state["rank_on"] = True
if st.session_state.get("rank_on"):
    with st.spinner("Berechne die Fehlerkurven über 5 Datensätze..."):
        rank_rows = _rank(data_params[:8], settings)
    st.plotly_chart(build_rank(rank_rows), width="stretch", key="rank_chart")
    st.table({"Wahre Neuronenzahl": [r["m"] for r in rank_rows], "gewählte k in den 5 Datensätzen": [", ".join(str(k) for k in r["chosen"]) for r in rank_rows], "Anteil richtig": [f"{r['correct']:.0%}" for r in rank_rows]})
    st.caption(f"Relativer Fehler der NMF (Mittel über 5 Datensätze, {max(n_el, 2)} Elektroden) je Komponentenzahl; senkrechte Striche: wahre Zahl. Der Knick ist der erste Punkt, an dem ein weiteres Element weniger als "
               f"{C.RANK_RATIO:.0%} dessen spart, was das letzte gebracht hat. Er trifft, solange jedes Neuron den Fehler sichtbar senkt - ein kleines Neuron fällt durch.")

st.markdown("---")

# --- Initialisierung ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Initialisierung und lokale Minima")
if st.button("Zufallsstarts und NNDSVD vergleichen (dauert einige Sekunden)", key="init_start"):
    st.session_state["init_on"] = True
if st.session_state.get("init_on"):
    with st.spinner("Rechne 8 Zufallsstarts und NNDSVD auf 5 Datensätzen..."):
        init_res = _init(data_params[:8], settings)
    st.plotly_chart(build_init(init_res["per_seed"]), width="stretch", key="init_chart")
    st.table({
        "Kennzahl (Mittel über 5 Datensätze)": ["Spitzen-F1 der Zufallsstarts: Mittel", "Streuung (Standardabweichung) über die Starts", "schlechtester / bester Start", "Start mit dem kleinsten Fehler", "Korrelation Fehler ↔ F1 über die Starts", "NNDSVD"],
        "Wert": [f"{init_res['mean']:.2f}", f"{init_res['std']:.2f}", f"{init_res['worst']:.2f} / {init_res['best']:.2f}", f"{init_res['lowest']:.2f}", f"{init_res['corr']:.2f}", f"{init_res['nndsvd']:.2f} (Fehler {init_res['err_nndsvd']:.3f} gegen {init_res['err_mean']:.3f} im Mittel der Zufallsstarts)"],
    })
    st.caption("Die multiplikativen Updates konvergieren gegen ein lokales Minimum; verschiedene Zufallsstarts landen bei etwas verschiedenen Zerlegungen. Eine negative Korrelation zwischen Fehler und F1 heißt: ein kleinerer Fehler ist (hier) meist auch besser getrennt - deshalb zählt der "
               "Start mit dem kleinsten Fehler. Punkte: Zufallsstarts (eine Farbe je Aufnahme), Rauten: NNDSVD.")

st.markdown("---")

# --- Szenen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer trennt was: acht Szenen im Vergleich")
if st.button("Acht Szenen vergleichen (dauert etwa eine Minute)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 8 Szenen × 5 Datensätze × 4 Verfahren..."):
        scene_rows = _scenes(data_params[:8], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "NMF": [f"{r['nmf']:.2f} ({r['nmf_min']:.2f}-{r['nmf_max']:.2f})" for r in scene_rows],
        "ICA": [f"{r['ica']:.2f}" for r in scene_rows],
        "SOBI": [f"{r['sobi']:.2f}" for r in scene_rows],
        "SCA": [f"{r['sca']:.2f}" for r in scene_rows],
        "Schwelle": [f"{r['baseline']:.2f}" for r in scene_rows],
    })
    st.caption("Szenen 2, 3 und 8 sind Stärken der NMF (abhängige Quellen, Rauschen gegenüber ICA und SOBI, eine Elektrode mit sehr verschiedenen Breiten), die übrigen Schwächen. Rauschen (sofern die Szene es nicht setzt) und Länge wie in der Seitenleiste, "
               "NMF-Regler wie eingestellt, bekannte Komponentenzahl. Mittel und Spanne über 5 feste Datensätze. Die Schwelle ist die Referenz für die Einkanal-Szenen (alle erkannten Spitzen jedem Neuron zugerechnet).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Daten sind nicht-negativ und die Muster addieren sich** | Bipolare Signale müssen gleichgerichtet werden; das gilt nur näherungsweise (der Nachschlag fällt weg oder wird mitgezählt, Überlappungen addieren sich nur in einer Phase). Rauschen wird mitgleichgerichtet: ohne Schwelle bildet es eine Grundlinie (Preset "Starkes Rauschen": 0.52 ohne, 0.74 mit Schwelle 3). | **ICA, SOBI, SCA** arbeiten auf den vorzeichenbehafteten Signalen |
| **Die Zerlegung ist eindeutig** | Sie ist es nicht: die NMF findet Zerlegungen mit **kleinerem** Fehler als die wahre (Standardfall: 0.013 gegen 0.049) und trennt trotzdem schlechter (0.77 gegen 1.00 bei ICA, SOBI, SCA); mehr Iterationen passen besser und trennen schlechter. Grund: die Mischspalten der Neuronen liegen dicht beieinander (Kosinus bis 0.90). | Sparsität (SCA; hier die L1-Strafe λ, die die Kosinus-Ähnlichkeit von 0.95 auf 0.98 hebt) |
| **Die Komponentenzahl ist bekannt** | Der Knick der Fehlerkurve trifft bei 2-4 Neuronen, bei 5 immer nur 4 (Preset "Fünf Neuronen, Komponentenzahl unbekannt"). | Modellwahl über Informationskriterien, Sparsität |
| **Mindestens so viele Elektroden wie Neuronen** | Bei zwei Elektroden und vier Neuronen 0.64 (ICA 0.36, SCA 0.97): der Rang begrenzt die Zerlegung. | **SCA** (mehr Quellen als Sensoren) |
| **Unabhängigkeit ist nicht nötig - aber Trennung braucht Verschiedenheit** | Bei synchronem Feuern gewinnt die NMF gegenüber ICA, SOBI und SCA (0.98 gegen 0.66 / 0.80 / 0.72 bei 90 % gemeinsamen Spikes), bei fast ununterscheidbaren Mustern nicht. | ICA, wenn die Quellen unabhängig sind |
| **Ein Kanal genügt** | Nur, wenn sich die Spektren der Neuronen unterscheiden: zwei Neuronen mit sehr verschiedenen Breiten 0.97, mit den gewohnten 0.50 (Schwelle 0.66). | Wellenform-Verfahren (Pipeline, Vorlagenabgleich) oder mehr Kanäle |
"""
)
st.caption("Die Nachbarn der Quellentrennung-Linie: ICA (Unabhängigkeit), SOBI (zeitliche Struktur), SCA (Sparsität, mehr Quellen als Sensoren), Spike-Sorting-Pipeline, Vorlagenabgleich und Verzögerungsgraph (Wellenform bzw. Zeitverzögerungen); die NMF lockert die ICA-Annahme zu Nicht-Negativität.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Elektrodensignale $x(t) = A\,s(t) + \varepsilon(t)$ mit positiver Mischmatrix $A \in \mathbb{R}_{\ge 0}^{n\times m}$ (hier $\propto 1/(d^2+\epsilon)$) und Quellen $s_i$ (Spikefolgen mit biphasischer Wellenform).

**Nicht-negative Darstellung.** $V_{jt} = \max\{-(x_{jt}-\tilde x_j) - c\,\hat\sigma_j,\ 0\}$ ($\tilde x_j$ Median, $\hat\sigma_j$ = MAD/0.6745, $c$ Schwelle) bzw. mit $|x_{jt}-\tilde x_j|$; ein Kanal: $V_{ft} = |\mathrm{STFT}(x)|_{f,t}$ (Hann-Fenster 32, Sprung 4).
Näherung $V \approx W H$ mit $W \ge 0$, $H \ge 0$, $W \in \mathbb{R}^{F\times k}$, $H \in \mathbb{R}^{k\times T}$.

**NMF.** $\min_{W,H\ge 0} \tfrac12\lVert V - WH\rVert_F^2 + \lambda\,\bar V \sum_{it} H_{it}$ (mit $\bar V$ = Mittelwert von $V$; $\lambda = 0$ klassisch). Multiplikative Updates (Lee und Seung):
$H \leftarrow H \odot \dfrac{W^\top V}{W^\top W H + \lambda\bar V}$, $W \leftarrow W \odot \dfrac{V H^\top}{W H H^\top}$; nach jeder Iteration werden die Spalten von $W$ auf Summe 1 normiert (Skalierung in $H$). Der Fehler sinkt monoton (Lee und Seung), die Zerlegung ist nicht eindeutig:
mit positiver invertierbarer $D$ ist auch $(WD, D^{-1}H)$ eine Zerlegung, wenn beide nicht-negativ bleiben. **NNDSVD:** aus dem führenden Singulärpaar und den positiven bzw. negativen Anteilen der übrigen.

**Komponentenzahl.** Fehlerkurve $e(k) = \lVert V - W_kH_k\rVert/\lVert V\rVert$; $k^\ast$ = kleinstes $k\ge 2$ mit $e(k)-e(k+1) < 0.25\,(e(k-1)-e(k))$.

**Bewertung.** Komponenten $H_i$ und wahre Neuronen $s_j$: Zuordnung per $|\text{Korrelation}|$ (Bitmasken-DP), Vorzeichen und Skala per Regression, Spitzen der Spur (tiefste zuerst, Mindestabstand 15, Schwelle $\max(4\sigma_{\text{MAD}},\,0.3\,\text{Tiefe})$), Treffer ±4 Abtastwerte, F1 je Neuron, gemittelt.
**Mischspalten-Kosinus:** mittlerer $|\cos|$ zwischen wahren Spalten $a_j$ und zugeordneten $W_i$. **Vergleichsfehler:** $\lVert V - A_{\text{wahr}} H^\ast\rVert/\lVert V\rVert$ mit $H^\ast$ per Updates bei festem $W = A$.

**Grenzen.** (1) Gleichrichtung ist nur näherungsweise additiv. (2) Nicht-Eindeutigkeit: kleinerer Fehler heißt nicht richtiger. (3) Rang $\le \min(n, T)$. (4) Bei einem Kanal entscheiden die Spektren.

Implementiert in `nm_algorithm.py` (Gleichrichtung, Spektrogramm, NMF, NNDSVD, Knick), `nm_ica.py`, `nm_sobi.py`, `nm_sca.py` (Vergleichsverfahren, wortgleich aus den Vorgänger-Demos), `nm_scenario.py` (Mehrelektroden-Generator mit Synchronität),
`nm_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Tabellen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
