"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe tm_presets.py in template-matching-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import nm_algorithm as alg
import nm_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_neurons_slider": SettingSpec("m", int, C.DEFAULT_N_NEURONS, C.N_NEURONS_MIN, C.N_NEURONS_MAX),
    "n_electrodes_slider": SettingSpec("n", int, C.DEFAULT_N_ELECTRODES, C.N_ELECTRODES_MIN, C.N_ELECTRODES_MAX),
    "rate_slider": SettingSpec("rate", float, C.DEFAULT_RATE_SCALE, C.RATE_SCALE_MIN, C.RATE_SCALE_MAX),
    "similarity_slider": SettingSpec("sim", float, C.DEFAULT_SIMILARITY, C.SIMILARITY_MIN, C.SIMILARITY_MAX),
    "jitter_slider": SettingSpec("jitter", float, C.DEFAULT_JITTER, C.JITTER_MIN, C.JITTER_MAX),
    "synchrony_slider": SettingSpec("sync", float, C.DEFAULT_SYNCHRONY, C.SYNCHRONY_MIN, C.SYNCHRONY_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "n_samples_slider": SettingSpec("T", int, C.DEFAULT_N_SAMPLES, C.N_SAMPLES_MIN, C.N_SAMPLES_MAX),
    "mode_select": SettingSpec("mode", _choice(alg.RECTIFY_MODES), C.DEFAULT_MODE),
    "threshold_slider": SettingSpec("thr", float, C.DEFAULT_THRESHOLD, C.THRESHOLD_MIN, C.THRESHOLD_MAX),
    "k_mode_select": SettingSpec("kmode", _choice(C.K_MODES), C.DEFAULT_K_MODE),
    "init_select": SettingSpec("init", _choice(alg.INITS), C.DEFAULT_INIT),
    "sparsity_slider": SettingSpec("lam", float, C.DEFAULT_SPARSITY, C.SPARSITY_MIN, C.SPARSITY_MAX),
    "iterations_slider": SettingSpec("iters", int, C.DEFAULT_ITERATIONS, C.ITERATIONS_MIN, C.ITERATIONS_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"m": "n_neurons_slider", "n": "n_electrodes_slider", "rate_scale": "rate_slider", "similarity": "similarity_slider", "jitter": "jitter_slider", "synchrony": "synchrony_slider",
               "noise": "noise_slider", "n_samples": "n_samples_slider", "mode": "mode_select", "threshold": "threshold_slider", "k_mode": "k_mode_select", "init": "init_select",
               "sparsity": "sparsity_slider", "iterations": "iterations_slider", "seed": "seed_input"}
KEPT = {"mode_select": "_mode_kept", "threshold_slider": "_threshold_kept"}        # bei einer Elektrode (Spektrogramm statt Gleichrichtung) ausgeblendet: der Wert bleibt hier erhalten


def init_session_state_defaults():
    """Fehlende Zustände auffüllen; die beiden Regler der Gleichrichtung (bei einer Elektrode ausgeblendet, sonst vom Widget-Zustand gelöscht) kehren zum zuletzt gewählten Wert zurück."""
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("rate_slider", 4), ("similarity_slider", 20), ("jitter_slider", 20), ("synchrony_slider", 20), ("threshold_slider", 2), ("sparsity_slider", 2)):
        st.session_state[key] = round(st.session_state.get(key, SETTING_SPECS[key].default) * step) / step
    st.session_state["iterations_slider"] = int(round(st.session_state.get("iterations_slider", C.DEFAULT_ITERATIONS) / 10) * 10)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
    st.session_state["_mode_kept"] = C.PRESETS[name]["mode"]
    st.session_state["_threshold_kept"] = C.PRESETS[name]["threshold"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
