"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nm_constants as C
from nm_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
EXPECTED_KIND = {"Vier Neuronen, vier Elektroden": "warning", "Synchrones Feuern": "success", "Starkes Rauschen": "warning", "Eine Elektrode, sehr verschiedene Breiten": "success",
                 "Eine Elektrode, Breiten wie gewohnt": "warning", "Fünf Neuronen, Komponentenzahl unbekannt": "warning"}


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox)}


def test_default_renders_without_exception():
    at = _run()
    assert any("NMF in Aktion" in m.value for m in at.markdown)
    assert not at.error and len(at.warning) == 1 and not at.success                   # Standardfall: ein anderes Verfahren trennt besser


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert (len(at.success) == 1 and not at.warning) if kind == "success" else (len(at.warning) == 1 and not at.success)


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MIN
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MIN
        at.session_state["threshold_slider"] = C.THRESHOLD_MAX
        at.session_state["iterations_slider"] = C.ITERATIONS_MIN
    _run(small)

    def large(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MAX
        at.session_state["n_electrodes_slider"] = C.N_ELECTRODES_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MAX
        at.session_state["similarity_slider"] = C.SIMILARITY_MAX
        at.session_state["jitter_slider"] = C.JITTER_MAX
        at.session_state["synchrony_slider"] = C.SYNCHRONY_MAX
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MAX
        at.session_state["sparsity_slider"] = C.SPARSITY_MAX
        at.session_state["mode_select"] = "absolute"
        at.session_state["init_select"] = "nndsvd"
        at.session_state["k_mode_select"] = "unknown"
    _run(large)

    def few_electrodes(at):
        at.session_state["n_neurons_slider"] = C.N_NEURONS_MAX
        at.session_state["n_electrodes_slider"] = 2
    _run(few_electrodes)


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("n", [1, 4])
def test_every_step_renders(step, n):
    def setup(at):
        at.session_state["n_electrodes_slider"] = n
        at.session_state["nm_step"] = step
    _run(setup)


def test_no_spikes_detected_renders():
    def setup(at):
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MIN
        at.session_state["n_neurons_slider"] = 2
        at.session_state["threshold_slider"] = C.THRESHOLD_MAX
        at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["nm_step"] = step
        at.run()
        assert not at.exception


def test_mode_and_threshold_are_hidden_with_one_electrode_and_their_values_are_kept():
    at = _run(lambda a: (a.session_state.__setitem__("mode_select", "absolute"), a.session_state.__setitem__("threshold_slider", 3.5)))
    assert {"Gleichrichtung", "Rauschschwelle c (Rausch-σ)"} <= _labels(at)
    at.session_state["n_electrodes_slider"] = 1
    at.run()
    assert not at.exception and not ({"Gleichrichtung", "Rauschschwelle c (Rausch-σ)"} & _labels(at))            # bei einer Elektrode: Spektrogramm, keine Gleichrichtung
    at.session_state["n_electrodes_slider"] = 4
    at.run()
    assert not at.exception
    assert [s for s in at.sidebar.selectbox if s.label == "Gleichrichtung"][0].value == "absolute"
    assert [s for s in at.sidebar.slider if s.label.startswith("Rauschschwelle")][0].value == 3.5


def test_threshold_sweep_option_disappears_with_one_electrode():
    at = _run()
    [s for s in at.selectbox if s.key == "sweep_select"][0].select("threshold")
    at.run()
    at.session_state["n_electrodes_slider"] = 1
    at.run()
    assert not at.exception and [s for s in at.selectbox if s.key == "sweep_select"][0].value != "threshold"


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["nm_step"] = 4
    at.run()
    assert not at.exception and at.session_state["nm_step"] == 4
    at.session_state["noise_slider"] = 0.2
    at.run()
    assert not at.exception and at.session_state["nm_step"] == 1


def test_window_start_is_clamped_when_the_recording_gets_shorter():
    at = _run(lambda a: a.session_state.__setitem__("window_start", 1500))
    at.session_state["n_samples_slider"] = C.N_SAMPLES_MIN
    at.run()
    assert not at.exception and at.session_state["window_start"] == 0


def test_experiments_run_on_demand_and_a_sweep_runs():
    at = _run()
    for parameter in ("n", "synchrony", "sparsity"):
        [s for s in at.selectbox if s.key == "sweep_select"][0].select(parameter)
        at.run()
        [b for b in at.button if b.key == "sweep_start"][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    for key in ("rank_start", "init_start"):
        [b for b in at.button if b.key == key][0].click()
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["rank_on"] and at.session_state["init_on"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "nm_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    calls = re.findall(r"plotly_chart\(", source)
    keyed = re.findall(r"plotly_chart\(.*?key=\"[a-z_]+\"\)", source)
    assert len(calls) == len(keyed) >= 14
    chart_keys = re.findall(r"plotly_chart\(.*?key=\"([a-z_]+)\"\)", source)
    assert len(set(chart_keys)) == len(chart_keys)                                                   # keine doppelten Schlüssel


def test_app_has_no_dead_file_links_and_the_verbatim_footer():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert not re.findall(r"\]\([a-z_]+\.py\)", source)
    assert "https://sebastianhanisch.net/kontakt.html" in source and "Interesse an einer maßgeschneiderten Lösung für" in source
