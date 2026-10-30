from __future__ import annotations

import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None


st.set_page_config(page_title="Beam Analysis Calculator", page_icon="📐", layout="wide")

PRESETS: dict[str, dict[str, Any]] = {
    "Simply supported · center point load": {
        "length": 6.0,
        "supports": [{"Type": "Pinned", "Location": 0.0}, {"Type": "Roller", "Location": 6.0}],
        "loads": [{"Type": "Point", "Start": 3.0, "End": 3.0, "Magnitude": 30.0, "End magnitude": 30.0}],
    },
    "Simply supported · full-span UDL": {
        "length": 6.0,
        "supports": [{"Type": "Pinned", "Location": 0.0}, {"Type": "Roller", "Location": 6.0}],
        "loads": [{"Type": "Distributed", "Start": 0.0, "End": 6.0, "Magnitude": 10.0, "End magnitude": 10.0}],
    },
    "Cantilever · tip point load": {
        "length": 3.0,
        "supports": [{"Type": "Fixed", "Location": 0.0}],
        "loads": [{"Type": "Point", "Start": 3.0, "End": 3.0, "Magnitude": 10.0, "End magnitude": 10.0}],
    },
    "Fixed · full-span UDL": {
        "length": 6.0,
        "supports": [{"Type": "Fixed", "Location": 0.0}, {"Type": "Fixed", "Location": 6.0}],
        "loads": [{"Type": "Distributed", "Start": 0.0, "End": 6.0, "Magnitude": 10.0, "End magnitude": 10.0}],
    },
    "Continuous · two-span UDL": {
        "length": 12.0,
        "supports": [{"Type": "Pinned", "Location": 0.0}, {"Type": "Roller", "Location": 6.0}, {"Type": "Roller", "Location": 12.0}],
        "loads": [{"Type": "Distributed", "Start": 0.0, "End": 12.0, "Magnitude": 10.0, "End magnitude": 10.0}],
    },
    "Overhanging · loaded overhang": {
        "length": 8.0,
        "supports": [{"Type": "Pinned", "Location": 0.0}, {"Type": "Roller", "Location": 6.0}],
        "loads": [{"Type": "Distributed", "Start": 6.0, "End": 8.0, "Magnitude": 10.0, "End magnitude": 10.0}],
    },
}


def beam_element_stiffness(length: float, ei: float) -> np.ndarray:
    l = length
    return ei / l**3 * np.array(
        [[12, 6*l, -12, 6*l], [6*l, 4*l*l, -6*l, 2*l*l],
         [-12, -6*l, 12, -6*l], [6*l, 2*l*l, -6*l, 4*l*l]], dtype=float
    )


def shape_functions(xi: float, length: float) -> np.ndarray:
    return np.array([1 - 3*xi**2 + 2*xi**3,
                     length * (xi - 2*xi**2 + xi**3),
                     3*xi**2 - 2*xi**3,
                     length * (-xi**2 + xi**3)])


def solve_beam(length: float, supports: pd.DataFrame, loads: pd.DataFrame,
               e_kn_m2: float, inertia_m4: float, area_m2: float,
               c_top_m: float, c_bottom_m: float, subdivisions: int = 10) -> dict[str, Any]:
    points = [0.0, length]
    points.extend(float(v) for v in supports["Location"])
    for _, row in loads.iterrows():
        points.extend([float(row["Start"]), float(row["End"])])
    base = np.linspace(0.0, length, max(2, subdivisions * 8) + 1)
    nodes = np.unique(np.clip(np.r_[base, points], 0.0, length))
    ndof = 2 * len(nodes)
    k_global = np.zeros((ndof, ndof))
    f_global = np.zeros(ndof)

    # Four-point Gauss integration gives consistent nodal loads for uniform and ramped loads.
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(4)
    for i, (x1, x2) in enumerate(zip(nodes[:-1], nodes[1:])):
        le = x2 - x1
        dofs = np.array([2*i, 2*i+1, 2*i+2, 2*i+3])
        ke = beam_element_stiffness(le, e_kn_m2 * inertia_m4)
        k_global[np.ix_(dofs, dofs)] += ke
        fe = np.zeros(4)
        for _, row in loads.iterrows():
            if row["Type"] != "Distributed":
                continue
            a, b = float(row["Start"]), float(row["End"])
            lo, hi = max(x1, a), min(x2, b)
            if hi <= lo or b <= a:
                continue
            q0, q1 = float(row["Magnitude"]), float(row["End magnitude"])
            for gx, gw in zip(gauss_x, gauss_w):
                x = (lo + hi) / 2 + gx * (hi - lo) / 2
                q_down = q0 + (q1 - q0) * (x - a) / (b - a)
                xi = (x - x1) / le
                # Upward displacement is positive; entered downward loads are positive.
                fe += shape_functions(xi, le) * (-q_down) * (hi - lo) / 2 * gw
        f_global[dofs] += fe

    def node_index(x: float) -> int:
        return int(np.argmin(np.abs(nodes - x)))

    for _, row in loads.iterrows():
        if row["Type"] == "Point":
            f_global[2 * node_index(float(row["Start"]))] -= float(row["Magnitude"])
        elif row["Type"] == "Moment":
            # Positive applied moment is counter-clockwise.
            f_global[2 * node_index(float(row["Start"])) + 1] += float(row["Magnitude"])

    fixed: dict[int, float] = {}
    for _, row in supports.iterrows():
        ni = node_index(float(row["Location"]))
        fixed[2*ni] = 0.0
        if row["Type"] == "Fixed":
            fixed[2*ni + 1] = 0.0
    free = np.array([i for i in range(ndof) if i not in fixed], dtype=int)
    u = np.zeros(ndof)
    if len(free):
        try:
            u[free] = np.linalg.solve(k_global[np.ix_(free, free)], f_global[free])
        except np.linalg.LinAlgError as exc:
            raise ValueError("The support layout is unstable. Add enough restraints to prevent rigid movement.") from exc
    residual = k_global @ u - f_global
    reactions = []
    for _, row in supports.iterrows():
        ni = node_index(float(row["Location"]))
        reactions.append({"Type": row["Type"], "Location": float(row["Location"]),
                          "Vertical reaction": residual[2*ni], "Moment reaction": residual[2*ni+1] if row["Type"] == "Fixed" else 0.0})

    def q_at(x: float, row: pd.Series) -> float:
        a, b = float(row["Start"]), float(row["End"])
        if b <= a or x < a or x > b:
            return 0.0
        return float(row["Magnitude"]) + (float(row["End magnitude"]) - float(row["Magnitude"])) * (x-a)/(b-a)

    xs = np.linspace(0, length, max(401, len(nodes)*4))
    shear, moment, deflection, slope = [], [], [], []
    for x in xs:
        v = sum(r["Vertical reaction"] for r in reactions if r["Location"] <= x + 1e-10)
        m = sum(r["Vertical reaction"] * (x-r["Location"]) for r in reactions if r["Location"] <= x + 1e-10)
        m -= sum(r["Moment reaction"] for r in reactions if r["Location"] <= x + 1e-10)
        for _, row in loads.iterrows():
            if row["Type"] == "Point" and float(row["Start"]) <= x:
                p = float(row["Magnitude"])
                v -= p
                m -= p * (x-float(row["Start"]))
            elif row["Type"] == "Moment" and float(row["Start"]) <= x:
                m -= float(row["Magnitude"])
            elif row["Type"] == "Distributed":
                a, b = float(row["Start"]), float(row["End"])
                hi = min(x, b)
                if hi > a and b > a:
                    q0, q1 = float(row["Magnitude"]), float(row["End magnitude"])
                    slope_q = (q1-q0)/(b-a)
                    total = q0*(hi-a) + slope_q*(hi-a)**2/2
                    first = q0*((x-a)*(hi-a) - (hi-a)**2/2) + slope_q*((x-a)*(hi-a)**2/2 - (hi-a)**3/3)
                    v -= total
                    m -= first
        shear.append(v)
        moment.append(m)
        ei = e_kn_m2 * inertia_m4
        if ei > 0:
            ni = min(np.searchsorted(nodes, x, side="right") - 1, len(nodes)-2)
            ni = max(0, ni)
            le = nodes[ni+1] - nodes[ni]
            xi = (x-nodes[ni])/le
            d = u[2*ni:2*ni+4]
            N = shape_functions(xi, le)
            dN = np.array([(-6*xi+6*xi**2)/le, 1-4*xi+3*xi**2,
                           (6*xi-6*xi**2)/le, -2*xi+3*xi**2])
            deflection.append(float(N @ d))
            slope.append(float(dN @ d))
        else:
            deflection.append(np.nan)
            slope.append(np.nan)
    shear_a, moment_a = np.asarray(shear), np.asarray(moment)
    data = {"x": xs, "shear": shear_a, "moment": moment_a,
            "deflection": np.asarray(deflection), "slope": np.asarray(slope),
            "shear_stress": np.abs(shear_a) / area_m2 if area_m2 > 0 else np.full_like(xs, np.nan),
            "bend_stress_top": -moment_a * c_top_m / inertia_m4 if inertia_m4 > 0 else np.full_like(xs, np.nan),
            "bend_stress_bottom": moment_a * c_bottom_m / inertia_m4 if inertia_m4 > 0 else np.full_like(xs, np.nan)}
    return {"nodes": nodes, "displacements": u, "reactions": pd.DataFrame(reactions), "diagrams": data}


def preset_to_units(preset: dict[str, Any], length_factor: float, force_factor: float,
                    line_factor: float, moment_factor: float) -> dict[str, Any]:
    converted_loads = []
    for load in preset["loads"]:
        magnitude_factor = {"Point": force_factor, "Distributed": line_factor,
                            "Moment": moment_factor}[load["Type"]]
        converted_loads.append({**load, "Start": load["Start"] * length_factor,
                                "End": load["End"] * length_factor,
                                "Magnitude": load["Magnitude"] * magnitude_factor,
                                "End magnitude": load["End magnitude"] * line_factor})
    return {"length": preset["length"] * length_factor,
            "supports": [{**r, "Location": r["Location"] * length_factor} for r in preset["supports"]],
            "loads": converted_loads}


st.markdown("""
<style>
.stApp { background: #f5f7fb; color: #152238; }
[data-testid="stHeader"] { background: rgba(255,255,255,.94); }
[data-testid="stToolbar"] { visibility: hidden; }
.block-container { max-width: 1480px; padding-top: 1.1rem; padding-bottom: 4rem; }
html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
h1, h2, h3 { color: #13243c !important; letter-spacing: -.025em; }
h1 { font-size: 2.1rem !important; font-weight: 750 !important; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e3e8f0; }
[data-testid="stBaseButton-primary"] { background: #1769aa; border-color: #1769aa; }
[data-testid="stBaseButton-primary"]:hover { background: #10578f; border-color: #10578f; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #e3e8f0; border-radius: 10px; }
.site-nav { display:flex; align-items:center; justify-content:space-between; gap:20px; background:#fff; border:1px solid #e5eaf1; border-radius:12px; padding:14px 22px; margin:0 0 28px; box-shadow:0 2px 8px rgba(24,42,68,.035); }
.site-brand { color:#152b45; font-size:18px; font-weight:800; letter-spacing:-.03em; white-space:nowrap; }
.site-brand span { color:#1674b8; }
.site-links { display:flex; gap:25px; color:#53647a; font-size:13px; font-weight:600; }
.site-account { color:#1769aa; font-size:13px; font-weight:700; white-space:nowrap; }
.hero-kicker { color:#1769aa; text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:.12em; margin-bottom:8px; }
.hero-copy { color:#607087; font-size:15px; max-width:760px; margin-top:5px; margin-bottom:20px; }
.section-note { color:#68788d; font-size:13px; margin-top:-8px; margin-bottom:12px; }
.beam-preview { background:#fff; border:1px solid #e4eaf2; border-radius:12px; padding:16px 20px 8px; margin:12px 0 22px; }
.beam-preview-title { color:#273b54; font-size:14px; font-weight:700; margin:0 0 4px; }
</style>
<div class="site-nav">
  <div class="site-brand">OPTIMAL<span>BEAM</span></div>
  <div class="site-links"><span>Beam Calculator</span><span>2D Structural Analysis</span><span>Free Tools</span><span>Resources</span></div>
  <div class="site-account">Log in &nbsp;&nbsp; <span style="background:#1769aa;color:white;padding:9px 14px;border-radius:7px">Try for free</span></div>
</div>
<div class="hero-kicker">Free structural analysis tool</div>
<h1 style="margin:0">Beam Calculator</h1>
<div class="hero-copy">Analyze simply supported, cantilever, fixed and continuous beams. Calculate reactions, shear, moment, slope, deflection and stress diagrams.</div>
""", unsafe_allow_html=True)

unit_system = st.radio("Unit system", ["Metric", "Imperial"], horizontal=True, key="unit_system_selector")
if unit_system == "Metric":
    length_label, force_label, line_label, moment_label = "m", "kN", "kN/m", "kN·m"
    length_to_m, force_to_kn, line_to_kn_m, moment_to_kn_m = 1.0, 1.0, 1.0, 1.0
    e_default, e_label, e_to_kn_m2 = 200000.0, "MPa", 1000.0
    i_default, i_label, i_to_m4 = 100_000_000.0, "mm⁴", 1e-12
    a_default, a_label, a_to_m2 = 10000.0, "mm²", 1e-6
    c_default, c_label, c_to_m = 100.0, "mm", 1e-3
else:
    length_label, force_label, line_label, moment_label = "ft", "kip", "kip/ft", "kip·ft"
    length_to_m, force_to_kn, line_to_kn_m, moment_to_kn_m = 0.3048, 4.4482216153, 14.59390294, 1.3558179483
    e_default, e_label, e_to_kn_m2 = 29000.0, "ksi", 6894.757293
    i_default, i_label, i_to_m4 = 240.0, "in⁴", 0.0254**4
    a_default, a_label, a_to_m2 = 10.0, "in²", 0.0254**2
    c_default, c_label, c_to_m = 5.0, "in", 0.0254

default_model = PRESETS["Simply supported · center point load"]
if "beam_model" not in st.session_state:
    st.session_state.beam_model = {"length": default_model["length"], "supports": pd.DataFrame(default_model["supports"]), "loads": pd.DataFrame(default_model["loads"])}
    st.session_state.beam_model_units = "Metric"
elif st.session_state.get("beam_model_units") != unit_system:
    old_units = st.session_state.get("beam_model_units", "Metric")
    base = {"Metric": (1.0, 1.0, 1.0, 1.0),
            "Imperial": (0.3048, 4.4482216153, 14.59390294, 1.3558179483)}
    old_length, old_force, old_line, old_moment = base[old_units]
    new_length, new_force, new_line, new_moment = base[unit_system]
    model_copy = st.session_state.beam_model
    model_copy["length"] *= old_length / new_length
    model_copy["supports"]["Location"] *= old_length / new_length
    model_copy["loads"]["Start"] *= old_length / new_length
    model_copy["loads"]["End"] *= old_length / new_length
    for idx, row in model_copy["loads"].iterrows():
        factor = {"Point": old_force / new_force, "Distributed": old_line / new_line,
                  "Moment": old_moment / new_moment}.get(row["Type"], 1.0)
        model_copy["loads"].loc[idx, "Magnitude"] *= factor
        model_copy["loads"].loc[idx, "End magnitude"] *= old_line / new_line
    st.session_state.beam_model = model_copy
    st.session_state.beam_model_units = unit_system
    for widget_key in (f"supports_{old_units}", f"loads_{old_units}",
                       f"supports_{unit_system}", f"loads_{unit_system}",
                       f"beam_length_{old_units}", f"beam_length_{unit_system}"):
        st.session_state.pop(widget_key, None)
st.subheader("Example models")
preset_col, preset_button_col = st.columns([4, 1])
with preset_col:
    chosen_preset = st.selectbox("Start with a common beam case", ["(custom)", *PRESETS.keys()])
with preset_button_col:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    load_example = st.button("Load example", use_container_width=True)
if load_example and chosen_preset != "(custom)":
    factors = (1.0, 1.0, 1.0, 1.0) if unit_system == "Metric" else (3.280839895, 0.224808943, 0.068521766, 0.73756215)
    preset_model = preset_to_units(PRESETS[chosen_preset], *factors)
    st.session_state.beam_model = {"length": preset_model["length"], "supports": pd.DataFrame(preset_model["supports"]), "loads": pd.DataFrame(preset_model["loads"])}
    st.session_state.beam_model_units = unit_system
    for widget_key in (f"supports_{unit_system}", f"loads_{unit_system}", f"beam_length_{unit_system}"):
        st.session_state.pop(widget_key, None)
    st.session_state.analysis_result = None

with st.sidebar:
    st.subheader("Section properties")
    E_input = st.number_input(f"Elastic modulus E ({e_label})", min_value=0.001, value=float(e_default), step=float(e_default/100))
    I_input = st.number_input(f"Second moment I ({i_label})", min_value=0.000001, value=float(i_default), format="%.6g")
    A_input = st.number_input(f"Area A ({a_label})", min_value=0.000001, value=float(a_default), format="%.6g")
    c_top_input = st.number_input(f"Top fiber distance ({c_label})", min_value=0.0, value=float(c_default), format="%.6g")
    c_bottom_input = st.number_input(f"Bottom fiber distance ({c_label})", min_value=0.0, value=float(c_default), format="%.6g")
    st.caption("Uniform section and material properties are assumed along the beam.")

model = st.session_state.beam_model
st.subheader("Model Setup")
st.markdown('<div class="section-note">Set the span, supports, loads and section properties for this beam model.</div>', unsafe_allow_html=True)
top1, top2 = st.columns([1, 1])
with top1:
    model_name = st.text_input("Model name", value=st.session_state.get("model_name", "Untitled"), key="model_name")
with top2:
    beam_length = st.number_input(f"Beam length ({length_label})", min_value=0.01, value=float(model["length"]), step=0.5, key=f"beam_length_{unit_system}")
with st.container(border=True):
    st.subheader("Supports")
    supports = st.data_editor(model["supports"], num_rows="dynamic", use_container_width=True, key=f"supports_{unit_system}", column_config={"Type": st.column_config.SelectboxColumn(options=["Pinned", "Roller", "Fixed"], required=True), "Location": st.column_config.NumberColumn(min_value=0.0, max_value=float(beam_length), step=0.25, format="%.3f")})
with st.container(border=True):
    st.subheader("Applied loads")
    loads = st.data_editor(model["loads"], num_rows="dynamic", use_container_width=True, key=f"loads_{unit_system}", column_config={
        "Type": st.column_config.SelectboxColumn(options=["Point", "Distributed", "Moment"], required=True),
        "Start": st.column_config.NumberColumn(f"Start ({length_label})", min_value=0.0, max_value=float(beam_length), step=0.25, format="%.3f"),
        "End": st.column_config.NumberColumn(f"End ({length_label})", min_value=0.0, max_value=float(beam_length), step=0.25, format="%.3f"),
        "Magnitude": st.column_config.NumberColumn(f"Magnitude ({force_label} / {line_label} / {moment_label})", format="%.4g"),
        "End magnitude": st.column_config.NumberColumn(f"End magnitude ({line_label})", format="%.4g"),
    })
st.caption("Loads are entered downward as positive. For a distributed load, use the start and end intensities; positive applied moments are counter-clockwise.")

with st.container(border=True):
    st.subheader("Section Properties")
    st.markdown('<div class="section-note">Choose a material and cross-section. Geometric shapes calculate area, inertia and extreme-fiber distances automatically.</div>', unsafe_allow_html=True)
    material_defaults = {
        "Steel": (200000.0, 29000.0),
        "Aluminum": (69000.0, 10000.0),
        "Timber": (11000.0, 1600.0),
        "Concrete": (30000.0, 4350.0),
        "Custom": (float(e_default), float(e_default)),
    }
    section_head = st.columns([1.1, 1.1, 1.4])
    with section_head[0]:
        material = st.selectbox("Material", list(material_defaults), key="section_material")
    with section_head[1]:
        section_shape = st.selectbox("Section type", ["Custom properties", "Rectangle", "Solid circle", "I-section", "Rectangular hollow"], key="section_shape")
    with section_head[2]:
        section_name = st.text_input("Section name", value="Custom beam section", key="section_name")
    material_e = material_defaults[material][0 if unit_system == "Metric" else 1]
    E_input = st.number_input(f"Elastic modulus E ({e_label})", min_value=0.001,
                              value=float(material_e), step=max(float(material_e/100), 0.1),
                              key=f"section_E_{unit_system}_{material}")
    section_dim_unit = "mm" if unit_system == "Metric" else "in"
    section_error = None
    if section_shape == "Custom properties":
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            A_input = st.number_input(f"Area ({a_label})", min_value=0.000001, value=float(a_default), format="%.6g", key=f"section_A_{unit_system}")
        with p2:
            I_input = st.number_input(f"Moment of inertia ({i_label})", min_value=0.000001, value=float(i_default), format="%.6g", key=f"section_I_{unit_system}")
        with p3:
            c_top_input = st.number_input(f"y-top ({c_label})", min_value=0.0, value=float(c_default), format="%.6g", key=f"section_ctop_{unit_system}")
        with p4:
            c_bottom_input = st.number_input(f"y-bottom ({c_label})", min_value=0.0, value=float(c_default), format="%.6g", key=f"section_cbottom_{unit_system}")
    else:
        if section_shape == "Rectangle":
            a, b = st.columns(2)
            with a:
                width = st.number_input(f"Width ({section_dim_unit})", min_value=0.01, value=200.0 if unit_system == "Metric" else 8.0, key=f"rect_width_{unit_system}")
            with b:
                depth = st.number_input(f"Depth ({section_dim_unit})", min_value=0.01, value=300.0 if unit_system == "Metric" else 12.0, key=f"rect_depth_{unit_system}")
            A_geom = width * depth
            I_geom = width * depth**3 / 12
            c_geom = depth / 2
        elif section_shape == "Solid circle":
            diameter = st.number_input(f"Diameter ({section_dim_unit})", min_value=0.01, value=200.0 if unit_system == "Metric" else 8.0, key=f"circle_diameter_{unit_system}")
            A_geom = np.pi * diameter**2 / 4
            I_geom = np.pi * diameter**4 / 64
            c_geom = diameter / 2
        elif section_shape == "I-section":
            i1, i2, i3 = st.columns(3)
            with i1:
                flange_width = st.number_input(f"Flange width ({section_dim_unit})", min_value=0.01, value=200.0 if unit_system == "Metric" else 8.0, key=f"i_bf_{unit_system}")
            with i2:
                overall_depth = st.number_input(f"Overall depth ({section_dim_unit})", min_value=0.02, value=300.0 if unit_system == "Metric" else 12.0, key=f"i_h_{unit_system}")
            with i3:
                flange_thickness = st.number_input(f"Flange thickness ({section_dim_unit})", min_value=0.01, value=12.0 if unit_system == "Metric" else 0.5, key=f"i_tf_{unit_system}")
            web_thickness = st.number_input(f"Web thickness ({section_dim_unit})", min_value=0.01, value=8.0 if unit_system == "Metric" else 0.3, key=f"i_tw_{unit_system}")
            if 2*flange_thickness >= overall_depth or web_thickness >= flange_width:
                section_error = "I-section geometry needs depth greater than twice the flange thickness, and flange width greater than web thickness."
                A_geom, I_geom, c_geom = 0.0, 0.0, 0.0
            else:
                web_depth = overall_depth - 2*flange_thickness
                A_geom = 2*flange_width*flange_thickness + web_thickness*web_depth
                I_geom = 2*(flange_width*flange_thickness**3/12 + flange_width*flange_thickness*((overall_depth-flange_thickness)/2)**2) + web_thickness*web_depth**3/12
                c_geom = overall_depth/2
        else:  # Rectangular hollow section
            h1, h2, h3 = st.columns(3)
            with h1:
                outer_width = st.number_input(f"Outer width ({section_dim_unit})", min_value=0.02, value=200.0 if unit_system == "Metric" else 8.0, key=f"rhs_b_{unit_system}")
            with h2:
                outer_depth = st.number_input(f"Outer depth ({section_dim_unit})", min_value=0.02, value=300.0 if unit_system == "Metric" else 12.0, key=f"rhs_h_{unit_system}")
            with h3:
                wall = st.number_input(f"Wall thickness ({section_dim_unit})", min_value=0.01, value=10.0 if unit_system == "Metric" else 0.4, key=f"rhs_t_{unit_system}")
            if 2*wall >= min(outer_width, outer_depth):
                section_error = "Hollow section wall thickness must be less than half of the smaller outside dimension."
                A_geom, I_geom, c_geom = 0.0, 0.0, 0.0
            else:
                A_geom = outer_width*outer_depth - (outer_width-2*wall)*(outer_depth-2*wall)
                I_geom = (outer_width*outer_depth**3 - (outer_width-2*wall)*(outer_depth-2*wall)**3)/12
                c_geom = outer_depth/2
        A_input, I_input = A_geom, I_geom
        c_top_input = c_bottom_input = c_geom
        output_cols = st.columns(4)
        output_cols[0].metric(f"Area ({a_label})", f"{A_input:,.4g}")
        output_cols[1].metric(f"I ({i_label})", f"{I_input:,.4g}")
        output_cols[2].metric(f"y-top ({c_label})", f"{c_top_input:,.4g}")
        output_cols[3].metric(f"y-bottom ({c_label})", f"{c_bottom_input:,.4g}")
    st.caption(f"{section_name} is treated as a uniform section over the full beam length. Stress uses σ = M·y/I and average shear uses τ = V/A.")

def draw_beam_preview(span: float, support_rows: pd.DataFrame, load_rows: pd.DataFrame) -> str:
    left, right, beam_y = 58, 942, 82
    scale = (right - left) / max(span, 0.01)
    parts = [f'<svg viewBox="0 0 1000 155" width="100%" role="img" aria-label="Beam model preview">',
             '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#dc5d4b"/></marker></defs>',
             f'<line x1="{left}" y1="{beam_y}" x2="{right}" y2="{beam_y}" stroke="#344b66" stroke-width="8" stroke-linecap="round"/>',
             f'<text x="{left}" y="142" fill="#718096" font-size="13">0 {length_label}</text>',
             f'<text x="{right}" y="142" fill="#718096" font-size="13" text-anchor="end">{span:g} {length_label}</text>']
    for _, row in support_rows.dropna(subset=["Type", "Location"]).iterrows():
        x = left + float(row["Location"]) * scale
        kind = row["Type"]
        if kind == "Fixed":
            parts.append(f'<path d="M{x-8},{beam_y-15} L{x+8},{beam_y-15} L{x+8},{beam_y+16} L{x-8},{beam_y+16} Z" fill="#75869a"/>')
            parts.append(f'<path d="M{x-17},{beam_y-15} l-10,8 m10,0 l-10,8 m10,0 l-10,8" stroke="#75869a" stroke-width="2"/>')
        else:
            parts.append(f'<path d="M{x},{beam_y+5} l-15,22 h30 z" fill="#eff5fb" stroke="#66809b" stroke-width="2"/>')
            if kind == "Roller":
                parts.append(f'<circle cx="{x-7}" cy="{beam_y+31}" r="3" fill="#66809b"/><circle cx="{x+7}" cy="{beam_y+31}" r="3" fill="#66809b"/>')
        parts.append(f'<text x="{x}" y="{beam_y+51}" fill="#55677e" font-size="12" text-anchor="middle">{kind}</text>')
    for _, row in load_rows.dropna(subset=["Type", "Start", "Magnitude"]).iterrows():
        kind = row["Type"]
        x1 = left + float(row["Start"]) * scale
        if kind == "Point":
            parts.append(f'<line x1="{x1}" y1="22" x2="{x1}" y2="{beam_y-7}" stroke="#dc5d4b" stroke-width="2.5" marker-end="url(#arrow)"/>')
            parts.append(f'<text x="{x1}" y="17" text-anchor="middle" fill="#bd4939" font-size="12">{float(row["Magnitude"]):g} {force_label}</text>')
        elif kind == "Distributed":
            end_location = float(row["End"]) if pd.notna(row["End"]) else float(row["Start"])
            x2 = left + end_location * scale
            parts.append(f'<line x1="{x1}" y1="25" x2="{x2}" y2="25" stroke="#dc5d4b" stroke-width="2"/>')
            for px in np.linspace(x1, x2, max(2, int((x2-x1)/36)+1)):
                parts.append(f'<line x1="{px:.1f}" y1="29" x2="{px:.1f}" y2="{beam_y-7}" stroke="#dc5d4b" stroke-width="1.5" marker-end="url(#arrow)"/>')
        else:
            parts.append(f'<path d="M{x1-18},{beam_y-9} A19,19 0 1 1 {x1+18},{beam_y-9}" fill="none" stroke="#dc5d4b" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</svg>')
    return ''.join(parts)

with st.container(border=True):
    st.markdown('<div class="beam-preview-title">Beam model preview</div>', unsafe_allow_html=True)
    st.markdown(draw_beam_preview(float(beam_length), supports, loads), unsafe_allow_html=True)

if st.button("Run analysis", type="primary", use_container_width=True):
    try:
        supports = supports.dropna(subset=["Type", "Location"]).copy()
        loads = loads.dropna(subset=["Type", "Start", "Magnitude"]).copy()
        loads["End"] = loads["End"].fillna(loads["Start"])
        loads["End magnitude"] = loads["End magnitude"].fillna(loads["Magnitude"])
        if supports.empty:
            raise ValueError("Add at least one support.")
        if section_error:
            raise ValueError(section_error)
        if ((supports["Location"] < 0) | (supports["Location"] > beam_length)).any():
            raise ValueError("Every support must be on the beam.")
        if not loads.empty and ((loads["Start"] < 0) | (loads["End"] > beam_length) | (loads["End"] < loads["Start"])).any():
            raise ValueError("Load locations must be within the beam and end at or after start.")
        s_si = supports.copy()
        s_si["Location"] = s_si["Location"].astype(float) * length_to_m
        l_si = loads.copy()
        for col in ["Start", "End"]:
            l_si[col] = l_si[col].astype(float) * length_to_m
        for idx, row in l_si.iterrows():
            if row["Type"] == "Point":
                l_si.loc[idx, "Magnitude"] = float(row["Magnitude"]) * force_to_kn
            elif row["Type"] == "Distributed":
                l_si.loc[idx, "Magnitude"] = float(row["Magnitude"]) * line_to_kn_m
                l_si.loc[idx, "End magnitude"] = float(row["End magnitude"]) * line_to_kn_m
            elif row["Type"] == "Moment":
                l_si.loc[idx, "Magnitude"] = float(row["Magnitude"]) * moment_to_kn_m
        result = solve_beam(beam_length*length_to_m, s_si, l_si, E_input*e_to_kn_m2, I_input*i_to_m4, A_input*a_to_m2, c_top_input*c_to_m, c_bottom_input*c_to_m)
        st.session_state.analysis_result = result
        st.session_state.analysis_info = {"name": model_name, "length": beam_length, "units": unit_system, "supports": supports, "loads": loads}
        st.session_state.analysis_info = {
            "name": model_name, "length": beam_length, "units": unit_system,
            "supports": supports, "loads": loads,
            "section": {"name": section_name, "material": material, "shape": section_shape,
                        "E": E_input, "E units": e_label, "A": A_input, "A units": a_label,
                        "I": I_input, "I units": i_label, "y top": c_top_input,
                        "y bottom": c_bottom_input, "distance units": c_label},
        }
    except Exception as exc:
        st.error(str(exc))

if st.session_state.get("analysis_result"):
    result = st.session_state.analysis_result
    info = st.session_state.analysis_info
    d = result["diagrams"]
    st.divider()
    st.header("Analysis results")
    section_used = info["section"]
    st.caption(f"Section: {section_used['name']} · {section_used['material']} · {section_used['shape']} · E = {section_used['E']:,.5g} {section_used['E units']} · I = {section_used['I']:,.5g} {section_used['I units']}")
    if unit_system == "Metric":
        x_plot = d["x"]
        v_plot, m_plot = d["shear"], d["moment"]
        delta_plot, slope_plot = d["deflection"]*1000, d["slope"]*1000
        stress_plot = d["bend_stress_top"] / 1000
        tau_plot = d["shear_stress"] / 1000
        x_unit, v_unit, m_unit, delta_unit = "m", "kN", "kN·m", "mm"
        stress_unit = "MPa"
    else:
        x_plot = d["x"] / 0.3048
        v_plot, m_plot = d["shear"] / 4.4482216153, d["moment"] / 1.3558179483
        delta_plot, slope_plot = d["deflection"] / 0.0254, d["slope"]
        stress_plot = d["bend_stress_top"] / 6894.757293
        tau_plot = d["shear_stress"] / 6894.757293
        x_unit, v_unit, m_unit, delta_unit = "ft", "kip", "kip·ft", "in"
        stress_unit = "ksi"
    reactions = result["reactions"].copy()
    reactions["Location"] /= length_to_m
    reactions["Vertical reaction"] /= force_to_kn
    reactions["Moment reaction"] /= moment_to_kn_m
    reactions.columns = ["Support", f"Location ({length_label})", f"Vertical ({force_label})", f"Moment ({moment_label})"]
    st.subheader("Support reactions")
    st.dataframe(reactions.round(4), use_container_width=True, hide_index=True)
    metrics = st.columns(4)
    metrics[0].metric(f"Max |V| ({v_unit})", f"{np.nanmax(np.abs(v_plot)):.4g}")
    metrics[1].metric(f"Max |M| ({m_unit})", f"{np.nanmax(np.abs(m_plot)):.4g}")
    metrics[2].metric(f"Max |deflection| ({delta_unit})", f"{np.nanmax(np.abs(delta_plot)):.4g}")
    metrics[3].metric(f"Max |bending stress| ({stress_unit})", f"{np.nanmax(np.abs(stress_plot)):.4g}")

    def plot_diagram(title: str, y: np.ndarray, y_label: str, color: str) -> None:
        if go is None:
            st.subheader(title)
            chart_data = pd.DataFrame({"Position": x_plot, y_label: y})
            st.line_chart(chart_data, x="Position", y=y_label, height=350)
            return
        fig = go.Figure(go.Scatter(x=x_plot, y=y, mode="lines", line={"color": color, "width": 3}, fill="tozeroy", fillcolor=f"rgba(45, 125, 210, 0.14)"))
        fig.add_hline(y=0, line_color="#64748b", line_width=1)
        fig.update_layout(title=title, xaxis_title=f"Position ({x_unit})", yaxis_title=y_label, height=350, margin={"l": 20, "r": 20, "t": 55, "b": 30}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        plot_diagram("Shear force", v_plot, v_unit, "#1d4ed8")
        plot_diagram("Deflection", delta_plot, delta_unit, "#7c3aed")
        plot_diagram("Average shear stress", tau_plot, stress_unit, "#0891b2")
    with col_b:
        plot_diagram("Bending moment", m_plot, m_unit, "#ea580c")
        plot_diagram("Slope", slope_plot, "mrad" if unit_system == "Metric" else "rad", "#16a34a")
        plot_diagram("Bending stress (top fiber)", stress_plot, stress_unit, "#be123c")

    export_data = {"model": info["name"], "date": date.today().isoformat(), "units": info["units"],
                   "beam_length": float(info["length"]), "supports": info["supports"].to_dict(orient="records"),
                   "loads": info["loads"].to_dict(orient="records"), "reactions": reactions.to_dict(orient="records"),
                   "loads": info["loads"].to_dict(orient="records"), "section": info["section"],
                   "reactions": reactions.to_dict(orient="records"),
                   "diagrams": {"x_m": d["x"].tolist(), "shear_kN": d["shear"].tolist(), "moment_kNm": d["moment"].tolist(),
                                "deflection_m": d["deflection"].tolist(), "slope": d["slope"].tolist(), "shear_stress_kPa": d["shear_stress"].tolist(), "bending_stress_top_kPa": d["bend_stress_top"].tolist()}}
    st.download_button("Download complete model and results (JSON)", json.dumps(export_data, indent=2, allow_nan=True), file_name=f"{model_name.replace(' ', '_')}_beam_results.json", mime="application/json")
    st.caption("Linear elastic Euler–Bernoulli beam model with small deflections and uniform section properties. Verify engineering decisions independently; this tool does not perform code capacity checks.")
