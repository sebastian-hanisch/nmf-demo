"""Defaults, Slider-Grenzen und feste Szenario-Größen der NMF-Demo. Das Szenario ist das der Vorgänger-Demos (Neuronen, Wellenformen, Feuerraten, Mischung wortgleich, dazu Ähnlichkeit und Schwankung)."""

# --- Szenario (fest, wortgleich aus ica/sobi/sca-demo) ------------------------------------------------------------------------
SAMPLE_RATE = 10_000                       # Hz
WAVEFORM_LENGTH = 30                       # Abtastwerte je Spike
PEAK_INDEX = 8                             # Lage der negativen Spitze in der Wellenform
OVERSHOOT = 0.4                            # Höhe des positiven Nachschlags
REFRACTORY = 20                            # Abtastwerte (2 ms)
NEURON_SIGMAS = (2.0, 3.0, 2.5, 4.0, 3.5)          # Breite der Spitze je Neuron
SIGMA_CENTER = 3.0                         # Ähnlichkeit 0: alle Neuronen bekommen diese Breite
NEURON_RATES = (20.0, 28.0, 35.0, 24.0, 31.0)      # Feuerrate in Hz (bei Feuerraten-Faktor 1)
NEURON_AMPLITUDES = (1.0, 0.8, 1.2, 0.7, 0.9)      # Spitzenamplitude an der nächsten Elektrode
NEURON_POSITIONS = ((0.10, 0.30), (0.35, 0.20), (0.60, 0.35), (0.85, 0.25), (0.50, 0.55))   # Elektroden liegen bei y = 0, x in [0, 1]
DISTANCE_EPS = 0.05
ACTIVE_THRESHOLD = 0.1                     # ein Neuron gilt an einem Zeitpunkt als aktiv, wenn seine Wellenform dort mehr als 10 % der Spitzenhöhe erreicht

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_N_NEURONS = 4
N_NEURONS_MIN, N_NEURONS_MAX = 2, 5
DEFAULT_N_ELECTRODES = 4
N_ELECTRODES_MIN, N_ELECTRODES_MAX = 1, 8
DEFAULT_RATE_SCALE = 1.0
RATE_SCALE_MIN, RATE_SCALE_MAX = 0.25, 4.0
DEFAULT_SIMILARITY = 1.0
SIMILARITY_MIN, SIMILARITY_MAX = 0.0, 2.0                                # 1 = Wellenformen wie gewohnt, 0 = alle Neuronen haben dieselbe Form, 2 = die Breitenunterschiede doppelt so groß
DEFAULT_JITTER = 0.0
JITTER_MIN, JITTER_MAX = 0.0, 0.3                                        # Streuung der Spitzenhöhe je Spike (relativ)
DEFAULT_NOISE = 0.05
NOISE_MIN, NOISE_MAX = 0.0, 1.0
DEFAULT_N_SAMPLES = 20_000
N_SAMPLES_MIN, N_SAMPLES_MAX = 5_000, 40_000
CONTRASTS = ("logcosh", "exp", "cube")
CONTRAST_LABELS = {"logcosh": "log cosh (robust)", "exp": "Gauß-Ableitung (sehr robust)", "cube": "Kurtosis (u³)"}
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
INIT_STARTS = (1, 2, 3, 4, 5)
DEFAULT_INIT_START = 1
DEFAULT_SEED = 7

# --- Vergleichsverfahren (aus ica/sobi/sca-demo) ---------------------------------------------------------------------------------
RECONSTRUCTIONS = ("l1", "single")
DEFAULT_RECONSTRUCTION = "l1"
MAX_ITER = 200                             # FastICA
TOL = 1e-6
JD_MAX_SWEEPS = 100                        # SOBI: Jacobi-Sweeps
JD_TOL = 1e-8
SOBI_LAGS = tuple(range(2, 21, 2))         # Verzögerungen des SOBI-Vergleichs
N_RESTARTS = 10                            # Neustarts (SCA-Vergleich: Achsen-k-Means)
KMEANS_MAX_ITER = 100

# --- NMF ---------------------------------------------------------------------------------------------------------------------------
NMF_MAX_ITER = 300
NMF_TOL = 1e-5                             # Abbruch, wenn der relative Fehler um weniger als tol (relativ) sinkt
NMF_RESTARTS = 5                           # Neustarts bei zufälligem Start
RANK_RATIO = 0.25                          # Knick-Kriterium der Komponentenzahl (siehe nm_algorithm.choose_k)
NMF_K_MAX = 8
SPECTROGRAM_WINDOW = 32                    # Einkanal-Szene: Fensterlänge und Sprung der Kurzzeit-Fouriertransformation (Abtastwerte)
SPECTROGRAM_HOP = 4
DEFAULT_SYNCHRONY = 0.0
SYNCHRONY_MIN, SYNCHRONY_MAX = 0.0, 0.9                                  # Anteil der Spikes, den ein Neuron mit Neuron 1 teilt (0 = unabhängig)
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed

# --- NMF-Regler ----------------------------------------------------------------------------------------------------------------------
DEFAULT_MODE = "negative"
DEFAULT_THRESHOLD = 2.0
THRESHOLD_MIN, THRESHOLD_MAX = 0.0, 4.0                                  # Rauschschwelle der Gleichrichtung in Rausch-Standardabweichungen
K_MODES = ("known", "unknown")
K_MODE_LABELS = {"known": "bekannt (= Neuronenzahl)", "unknown": "unbekannt (Knick der Fehlerkurve)"}
DEFAULT_K_MODE = "known"
DEFAULT_INIT = "random"
DEFAULT_SPARSITY = 0.0
SPARSITY_MIN, SPARSITY_MAX = 0.0, 3.0
DEFAULT_ITERATIONS = NMF_MAX_ITER
ITERATIONS_MIN, ITERATIONS_MAX = 20, 1000


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_NEURONS, n=DEFAULT_N_ELECTRODES, rate_scale=DEFAULT_RATE_SCALE, similarity=DEFAULT_SIMILARITY, jitter=DEFAULT_JITTER, noise=DEFAULT_NOISE, n_samples=DEFAULT_N_SAMPLES,
                synchrony=DEFAULT_SYNCHRONY, mode=DEFAULT_MODE, threshold=DEFAULT_THRESHOLD, k_mode=DEFAULT_K_MODE, init=DEFAULT_INIT, sparsity=DEFAULT_SPARSITY, iterations=DEFAULT_ITERATIONS,
                seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Vier Neuronen, vier Elektroden": _preset(),
    "Synchrones Feuern": _preset(synchrony=0.9),
    "Starkes Rauschen": _preset(noise=1.0, threshold=3.0),
    "Eine Elektrode, sehr verschiedene Breiten": _preset(m=2, n=1, similarity=2.0),
    "Eine Elektrode, Breiten wie gewohnt": _preset(m=2, n=1),
    "Fünf Neuronen, Komponentenzahl unbekannt": _preset(m=5, n=6, k_mode="unknown"),
}
PRESET_HELP = {
    "Vier Neuronen, vier Elektroden": "Der Standardfall der Linie: die NMF erreicht Spitzen-F1 0.77 (Mittel über fünf Aufnahmen), ICA, SOBI und SCA 1.00. Die Mischspalten der Neuronen liegen dicht beieinander (größte Kosinus-Ähnlichkeit 0.90), "
                                      "und die NMF findet eine Zerlegung, die die gleichgerichteten Daten besser erklärt (Fehler 0.013) als die wahre (0.049) - aber die falsche.",
    "Synchrones Feuern": "Im Mittel über fünf Aufnahmen: 90 % der Spikes der Neuronen 2-4 fallen mit Spikes von Neuron 1 zusammen: die Quellen sind nicht mehr unabhängig. ICA (0.66), SOBI (0.80) und SCA (0.72) verlieren, die NMF bleibt bei 0.98 (Korrelation 0.87 gegen 0.56 / 0.68 / 0.64). "
                         "Bedenken: bei so viel Gleichzeitigkeit ist die Aufgabe leicht - die beste Einzelelektrode erreicht 0.92 (Korrelation 0.94).",
    "Starkes Rauschen": "Im Mittel über fünf Aufnahmen: Rauschen von 100 % des Neuronen-Signals. Ohne Schwelle in der Gleichrichtung bildet das Rauschen eine positive Grundlinie und die NMF fällt auf 0.52; mit Schwelle 3 Rausch-σ sind es 0.74 - mehr als ICA (0.43) und SOBI (0.38), "
                        "aber weniger als SCA (0.86).",
    "Eine Elektrode, sehr verschiedene Breiten": "Im Mittel über fünf Aufnahmen (in dieser Aufnahme 0.89): nur eine Elektrode, zwei Neuronen mit sehr verschieden breiten Spitzen (Ähnlichkeit 2: Breite 1 gegen 3): die NMF zerlegt das Spektrogramm und trennt sie mit Spitzen-F1 0.97 - die Schwelle auf der Elektrode erreicht 0.66, "
                                                "ICA und SOBI 0.33, SCA 0.04 (sie brauchen mehrere Elektroden).",
    "Eine Elektrode, Breiten wie gewohnt": "Dieselbe Einkanal-Aufnahme mit den gewohnten Breiten (2 gegen 3): die kurzen Spitzen haben fast dieselben Spektren, die NMF kommt im Mittel über fünf Aufnahmen auf 0.50 (in dieser Aufnahme 0.61) und liegt unter der Schwelle (0.66 bzw. 0.65) - sie findet die Spikes, "
                                           "ordnet sie aber kaum den Neuronen zu.",
    "Fünf Neuronen, Komponentenzahl unbekannt": "Im Mittel über fünf Aufnahmen: die Komponentenzahl wird am Knick der Fehlerkurve gewählt: bei 2, 3 und 4 Neuronen trifft das in jeder der fünf Aufnahmen, bei fünf immer nur vier - das fünfte Neuron ist zu klein, um den Fehler sichtbar zu senken. "
                                                "Spitzen-F1 0.51 statt 0.61 mit richtiger Zahl; ICA 0.86, SCA 0.99.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Spitzen-F1 der NMF (f1), Komponentenzahl (k), erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Vier Neuronen, vier Elektroden": {"f1": (0.6, 0.9), "k": (4, 4), "verdict": ("others_win",)},
    "Synchrones Feuern": {"f1": (0.9, 1.0), "verdict": ("nmf_wins",)},
    "Starkes Rauschen": {"f1": (0.6, 0.85), "verdict": ("others_win",)},
    "Eine Elektrode, sehr verschiedene Breiten": {"f1": (0.75, 1.0), "verdict": ("single_works",)},
    "Eine Elektrode, Breiten wie gewohnt": {"f1": (0.35, 0.75), "verdict": ("single_weak",)},
    "Fünf Neuronen, Komponentenzahl unbekannt": {"f1": (0.3, 0.7), "k": (4, 4), "verdict": ("wrong_k",)},
}
