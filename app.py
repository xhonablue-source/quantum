# Final Combined MathCraft: Quantum Quest App

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import math
from typing import Tuple, Dict, List
import time
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# MathCraft: Quantum Quest — Enhanced Interactive Quantum Mechanics
# Comprehensive Streamlit app combining story mode, labs, and visualizations
# -----------------------------------------------------------------------------

# --- Page Configuration ---
st.set_page_config(
    page_title="MathCraft: Quantum Quest",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for persistence
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'completed_modules' not in st.session_state:
    st.session_state.completed_modules = set()
if 'interference_challenges' not in st.session_state:
    st.session_state.interference_challenges = {
        "max_constructive": False,
        "max_destructive": False,
        "half_prob": False
    }

# ----------------------- Utility Functions and Quantum Concepts ----------------------------
# This section defines the core logic for the quantum simulations and visualizations.

def ket0() -> np.ndarray:
    return np.array([[1.0], [0.0]], dtype=complex)

def ket1() -> np.ndarray:
    return np.array([[0.0], [1.0]], dtype=complex)

def normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    return state / norm if norm != 0 else state

# Quantum Gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

def Rx(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def apply(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    return gate @ state

def measure_probs(state: np.ndarray) -> Dict[str, float]:
    a = state[0,0]
    b = state[1,0]
    p0 = float(np.real(a*np.conj(a)))
    p1 = float(np.real(b*np.conj(b)))
    return {"0": p0, "1": p1}

def sample_measure(state: np.ndarray) -> str:
    probs = measure_probs(state)
    return np.random.choice(["0","1"], p=[probs["0"], probs["1"]])

def bloch_coords(state: np.ndarray) -> Tuple[float, float, float]:
    s = normalize(state)
    a = s[0,0]
    b = s[1,0]
    x = 2 * np.real(np.conj(a)*b)
    y = 2 * np.imag(np.conj(a)*b)
    z = float(np.real(np.conj(a)*a - np.conj(b)*b))
    return (float(x), float(y), float(z))

def cfmt(z: complex, precision: int = 3) -> str:
    real_part = f"{z.real:.{precision}f}" if abs(z.real) > 1e-10 else "0"
    imag_part = f"{abs(z.imag):.{precision}f}" if abs(z.imag) > 1e-10 else "0"
    
    if abs(z.imag) < 1e-10:
        return real_part
    elif abs(z.real) < 1e-10:
        sign = "+" if z.imag >= 0 else "-"
        return f"{sign[1:] if sign == '+' else sign}{imag_part}i"
    else:
        sign = "+" if z.imag >= 0 else "-"
        return f"{real_part} {sign} {imag_part}i"

def add_xp(points: int, reason: str = ""):
    st.session_state.xp += points
    if reason:
        st.success(f"+{points} XP: {reason}")
    
    new_level = (st.session_state.xp // 100) + 1
    if new_level > st.session_state.current_level:
        st.session_state.current_level = new_level
        st.balloons()
        st.success(f"🎉 Level Up! You're now Level {new_level}!")

def create_bloch_sphere(x: float, y: float, z: float, title: str = "Qubit State on Bloch Sphere") -> go.Figure:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    fig.add_trace(go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z,
        opacity=0.1, colorscale='Blues', showscale=False, hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='markers+lines',
        marker=dict(size=[0, 8], color=['blue', 'red']),
        line=dict(color='red', width=6),
        name='|ψ⟩'
    ))
    
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=2), name='X-axis', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='black', width=2), name='Y-axis', showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='black', width=2), name='Z-axis', showlegend=False))
    
    fig.add_trace(go.Scatter3d(x=[1.3], y=[0], z=[0], mode='text', text=['X'], textfont=dict(size=14, color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[1.3], z=[0], mode='text', text=['Y'], textfont=dict(size=14, color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1.3], mode='text', text=['|0⟩'], textfont=dict(size=14, color='blue'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-1.3], mode='text', text=['|1⟩'], textfont=dict(size=14, color='red'), showlegend=False))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            yaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            zaxis=dict(range=[-1.5, 1.5], showticklabels=False),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=600,
        height=500,
        showlegend=True
    )
    
    return fig

# ----------------------- Main App Structure ----------------------------
# This is the central function that runs the entire application and manages navigation.

def main_app():
    st.title("⚛️ MathCraft: Quantum Quest")
    st.caption("A comprehensive, story-driven introduction to quantum mechanics — Built by Xavier Honablue M.Ed for CognitiveCloud.ai")

    # --- Sidebar ---
    with st.sidebar:
        st.header("🎮 Player Dashboard")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XP", st.session_state.xp)
        with col2:
            st.metric("Level", st.session_state.current_level)
        current_level_xp = st.session_state.xp % 100
        progress = current_level_xp / 100
        st.progress(progress)
        st.caption(f"{current_level_xp}/100 XP to next level")
    
        if st.session_state.achievements:
            st.subheader("🏆 Achievements")
            for achievement in st.session_state.achievements[-3:]:
                st.success(f"🎖️ {achievement}")
    
        st.divider()
    
        st.header("🗺️ Navigate")
        page = st.radio(
            "Choose a module:",
            [
                "🏰 Story Mode",
                "🧪 Superposition Lab",
                "⚙️ Quantum Gates Workshop",
                "🌍 Bloch Explorer",
                "🌊 Interference Sandbox",
                "🎯 Stern–Gerlach Lab",
                "🔗 Entanglement Workshop",
                "📊 Quantum Simulator",
                "❓ Quizzes & Challenges",
                "🚀 Future Quest Modules",
                "📚 Resources"
            ]
        )

    # --- Page Content ---
    if page == "🏰 Story Mode":
        story_mode()
    elif page == "🧪 Superposition Lab":
        superposition_lab()
    elif page == "⚙️ Quantum Gates Workshop":
        quantum_gates_workshop()
    elif page == "🌍 Bloch Explorer":
        bloch_explorer()
    elif page == "🌊 Interference Sandbox":
        interference_sandbox()
    elif page == "🎯 Stern–Gerlach Lab":
        stern_gerlach_lab()
    elif page == "🔗 Entanglement Workshop":
        entanglement_workshop()
    elif page == "📊 Quantum Simulator":
        quantum_simulator()
    elif page == "❓ Quizzes & Challenges":
        quizzes_and_challenges()
    elif page == "🚀 Future Quest Modules":
        future_quest_modules()
    elif page == "📚 Resources":
        resources_page()

# ----------------------- Foundational Modules ----------------------------
# These modules correspond to the "Part 1: Foundational Labs" section.

def story_mode():
    st.header("Episode 1: The Mysterious Q-Box")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📖 The Discovery
        
        In the dusty attic of an old physics laboratory, you discover a peculiar device labeled **Q-BOX**. 
        A yellowed note attached reads:
        
        > *"Within this quantum realm lives a qubit - a mysterious entity that exists in multiple states 
        > simultaneously until observed. Master the art of quantum state preparation, and unlock the 
        > secrets of the quantum world."*
        
        Your mission begins now, quantum explorer! 🚀
        """)
    
        st.markdown("### 🎯 Mission 1: Create the Hadamard State")
        st.info("Target: Prepare the state (|0⟩ + |1⟩)/√2 with equal probability of measuring 0 or 1")
    
        theta = st.slider("🔄 Rotation around Y-axis (θ)", 0.0, float(np.pi), value=float(np.pi/2), step=0.01)
        phi = st.slider("🌀 Phase rotation around Z-axis (φ)", 0.0, float(2*np.pi), value=0.0, step=0.01)
    
        state = apply(Ry(theta), ket0())
        state = apply(Rz(phi), state)
        probs = measure_probs(state)
    
        st.markdown("### Current Quantum State")
        st.code(f"|ψ⟩ = {cfmt(state[0,0])}|0⟩ + {cfmt(state[1,0])}|1⟩")
    
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("P(|0⟩)", f"{probs['0']:.3f}")
        with col_b:
            st.metric("P(|1⟩)", f"{probs['1']:.3f}")
    
        trials = st.number_input("Number of measurements", min_value=10, max_value=1000, value=100, step=10)
    
        if st.button("🎲 Run Quantum Measurements", type="primary"):
            with st.spinner("Measuring quantum state..."):
                time.sleep(0.5)
                results = [sample_measure(state) for _ in range(int(trials))]
                p0_observed = results.count("0") / len(results)
                p1_observed = 1 - p0_observed
    
                st.success(f"📊 Results: |0⟩ → {p0_observed:.3f}, |1⟩ → {p1_observed:.3f}")
    
                if abs(p0_observed - 0.5) < 0.1 and abs(p1_observed - 0.5) < 0.1:
                    add_xp(15, "Successfully created superposition state!")
                    if "Hadamard Master" not in st.session_state.achievements:
                        st.session_state.achievements.append("Hadamard Master")
                        st.balloons()
                else:
                    add_xp(5, "Good attempt! Try adjusting θ to π/2 for perfect superposition.")
    
    with col2:
        st.markdown("### 🧠 Quantum Concepts")
        st.markdown("""
        **Superposition**: A qubit can exist in a combination of |0⟩ and |1⟩ states simultaneously.
        
        **Measurement**: Collapses the superposition, giving either |0⟩ or |1⟩ with probabilities |α|² and |β|².
        
        **Phase**: The relative phase between states affects interference but not individual measurement probabilities.
        """)
    
        x, y, z = bloch_coords(state)
        fig = create_bloch_sphere(x, y, z, "Your Qubit State")
        st.plotly_chart(fig, use_container_width=True)

def superposition_lab():
    st.header("Superposition Laboratory")
    st.markdown("Craft custom quantum states by directly setting complex amplitudes")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("State Designer")
    
        st.markdown("**α coefficient (for |0⟩):**")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            a_real = st.slider("Real(α)", -1.0, 1.0, 1.0, 0.01)
        with col_a2:
            a_imag = st.slider("Imag(α)", -1.0, 1.0, 0.0, 0.01)
    
        st.markdown("**β coefficient (for |1⟩):**")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            b_real = st.slider("Real(β)", -1.0, 1.0, 0.0, 0.01)
        with col_b2:
            b_imag = st.slider("Imag(β)", -1.0, 1.0, 0.0, 0.01)
    
        psi = np.array([[a_real + 1j*a_imag], [b_real + 1j*b_imag]], dtype=complex)
        psi = normalize(psi)
    
        st.markdown("### Normalized Quantum State")
        st.code(f"|ψ⟩ = {cfmt(psi[0,0])}|0⟩ + {cfmt(psi[1,0])}|1⟩")
    
        probs = measure_probs(psi)
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.metric("P(|0⟩)", f"{probs['0']:.4f}")
        with col_p2:
            st.metric("P(|1⟩)", f"{probs['1']:.4f}")
    
        x, y, z = bloch_coords(psi)
        st.markdown("### Bloch Sphere Coordinates")
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.metric("X", f"{x:.3f}")
        with col_y:
            st.metric("Y", f"{y:.3f}")
        with col_z:
            st.metric("Z", f"{z:.3f}")
    
        nshots = st.number_input("Number of measurements", 10, 2000, 500, 50)
    
        if st.button("🎲 Perform Measurements", type="primary"):
            with st.spinner("Running quantum measurements..."):
                results = [sample_measure(psi) for _ in range(int(nshots))]
                df = pd.DataFrame({'Outcome': results})
                counts = df['Outcome'].value_counts(normalize=True).sort_index()
    
                fig_hist = px.bar(
                    x=counts.index,
                    y=counts.values,
                    labels={'x': 'Measurement Outcome', 'y': 'Probability'},
                    title="Measurement Results"
                )
                fig_hist.update_traces(marker_color=['lightblue', 'lightcoral'])
                st.plotly_chart(fig_hist, use_container_width=True)
    
                add_xp(8, "Superposition experiment completed!")
    
    with col2:
        st.subheader("Visualization")
        x, y, z = bloch_coords(psi)
        fig = create_bloch_sphere(x, y, z)
        st.plotly_chart(fig, use_container_width=True)
    
        st.subheader("🎯 Challenges")
        st.markdown("""
        1. Create a state with **75%** chance of |0⟩
        2. Make a state on the **X-Y plane** (z=0)
        3. Create the state **(|0⟩ - i|1⟩)/√2**
        """)

def quantum_gates_workshop():
    st.header("Quantum Gates Workshop")
    st.markdown("Learn how quantum gates transform qubit states")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("1. Choose Initial State")
        init_state = st.selectbox("Starting state:", ["|0⟩", "|1⟩", "|+⟩ = (|0⟩+|1⟩)/√2", "|-⟩ = (|0⟩-|1⟩)/√2"])
    
        if init_state == "|0⟩":
            state = ket0()
        elif init_state == "|1⟩":
            state = ket1()
        elif init_state == "|+⟩ = (|0⟩+|1⟩)/√2":
            state = apply(H, ket0())
        else:
            state = apply(H, ket1())
    
        st.subheader("2. Apply Quantum Gates")
    
        gate_sequence = []
    
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            pauli_gate = st.selectbox("Pauli Gates:", ["None", "X (NOT)", "Y", "Z"])
            if pauli_gate != "None":
                gate_sequence.append(pauli_gate)
    
        with col_g2:
            had_gate = st.checkbox("Apply Hadamard (H)")
            if had_gate:
                gate_sequence.append("H")
    
        st.markdown("**Rotation Gates:**")
        col_r1, col_r2, col_r3 = st.columns(3)
    
        with col_r1:
            rx_angle = st.slider("Rx rotation (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            if rx_angle != 0:
                gate_sequence.append(f"Rx({rx_angle:.2f})")
    
        with col_r2:
            ry_angle = st.slider("Ry rotation (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            if ry_angle != 0:
                gate_sequence.append(f"Ry({ry_angle:.2f})")
    
        with col_r3:
            rz_angle = st.slider("Rz rotation (radians)", 0.0, 2*np.pi, 0.0, 0.1)
            if rz_angle != 0:
                gate_sequence.append(f"Rz({rz_angle:.2f})")
    
        phase_angle = st.slider("Phase gate φ", 0.0, 2*np.pi, 0.0, 0.1)
        if phase_angle != 0:
            gate_sequence.append(f"Phase({phase_angle:.2f})")
    
        final_state = state.copy()
        for gate in gate_sequence:
            if gate == "X (NOT)":
                final_state = apply(X, final_state)
            elif gate == "Y":
                final_state = apply(Y, final_state)
            elif gate == "Z":
                final_state = apply(Z, final_state)
            elif gate == "H":
                final_state = apply(H, final_state)
            elif gate.startswith("Rx"):
                final_state = apply(Rx(rx_angle), final_state)
            elif gate.startswith("Ry"):
                final_state = apply(Ry(ry_angle), final_state)
            elif gate.startswith("Rz"):
                final_state = apply(Rz(rz_angle), final_state)
            elif gate.startswith("Phase"):
                final_state = apply(phase(phase_angle), final_state)
    
        st.subheader("3. Results")
    
        if gate_sequence:
            st.markdown(f"**Applied Gates:** {' → '.join(gate_sequence)}")
        else:
            st.markdown("**No gates applied**")
    
        col_init, col_arrow, col_final = st.columns([2, 1, 2])
    
        with col_init:
            st.markdown("**Initial State:**")
            st.code(f"{init_state}")
            init_probs = measure_probs(state)
            st.write(f"P(0) = {init_probs['0']:.3f}")
            st.write(f"P(1) = {init_probs['1']:.3f}")
    
        with col_arrow:
            st.markdown(f"# ➡️")
    
        with col_final:
            st.markdown("**Final State:**")
            st.code(f"|ψ⟩ = {cfmt(final_state[0,0])}|0⟩ + {cfmt(final_state[1,0])}|1⟩")
            final_probs = measure_probs(final_state)
            st.write(f"P(0) = {final_probs['0']:.3f}")
            st.write(f"P(1) = {final_probs['1']:.3f}")
    
    with col2:
        x, y, z = bloch_coords(final_state)
        fig = create_bloch_sphere(x, y, z, "Final State on Bloch Sphere")
        st.plotly_chart(fig, use_container_width=True)
    
        st.subheader("🧠 Gate Functions")
        st.markdown("""
        - **X (Pauli-X)**: A quantum NOT gate. Flips |0⟩ to |1⟩ and vice versa. Corresponds to a 180° rotation around the X-axis.
        - **H (Hadamard)**: Creates a superposition. Rotates a state on the Z-axis (like |0⟩) to the X-Y plane (to |+⟩).
        - **Rx, Ry, Rz**: Generic rotation gates that let you rotate the qubit state by a specified angle around the X, Y, or Z axes.
        - **Phase**: Applies a relative phase shift to the |1⟩ state, rotating the state vector around the Z-axis.
        """)

def bloch_explorer():
    st.header("Bloch Explorer")
    st.markdown("Freely navigate the Bloch sphere and observe the state vector in real-time.")
    
    st.markdown("Use the sliders to control the polar ($θ$) and azimuthal ($φ$) angles.")
    
    theta_slider = st.slider("Polar Angle (θ) from Z-axis", 0.0, np.pi, np.pi/4, 0.01)
    phi_slider = st.slider("Azimuthal Angle (φ) from X-axis", 0.0, 2 * np.pi, np.pi/2, 0.01)
    
    # Calculate cartesian coordinates from spherical
    x = np.sin(theta_slider) * np.cos(phi_slider)
    y = np.sin(theta_slider) * np.sin(phi_slider)
    z = np.cos(theta_slider)
    
    # Calculate the quantum state vector
    alpha = np.cos(theta_slider/2)
    beta = np.sin(theta_slider/2) * np.exp(1j * phi_slider)
    state_vector = np.array([[alpha], [beta]], dtype=complex)
    
    # Display the Bloch sphere
    fig = create_bloch_sphere(x, y, z, "Real-time Bloch Sphere")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Current State Information")
    st.code(f"|ψ⟩ = {cfmt(state_vector[0,0])}|0⟩ + {cfmt(state_vector[1,0])}|1⟩")
    
    probs = measure_probs(state_vector)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("P(|0⟩)", f"{probs['0']:.3f}")
    with col2:
        st.metric("P(|1⟩)", f"{probs['1']:.3f}")
    
    st.markdown("---")
    st.markdown("### 🎯 Challenges")
    st.markdown("""
    1.  Find the coordinates for the **|1⟩** state.
    2.  Find the coordinates for the **|+⟩** state.
    3.  Find the coordinates for the state **(|0⟩ - i|1⟩)/√2**.
    
    *Hint: Think about which axes the state vector should align with.*
    """)

def interference_sandbox():
    st.header("Interference Sandbox")
    st.markdown("Explore how quantum states interfere, a key principle behind quantum algorithms.")

    st.write("Imagine two paths a particle can take. Each path has a different phase shift.")
    st.markdown("The final probability of a state is not just the sum of probabilities of each path, but a result of their interference.")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Two-Path Interference")
        st.write("Here, we have a simple interferometer model. The particle can take a Path 0 or Path 1.")
    
        # Path 0
        path0_amplitude = st.number_input("Path 0 Amplitude (real)", -1.0, 1.0, 1/np.sqrt(2), step=0.01, key="p0_amp")
        path0_phase = st.slider("Path 0 Phase (radians)", 0.0, 2*np.pi, 0.0, step=0.01, key="p0_phase")
    
        # Path 1
        path1_amplitude = st.number_input("Path 1 Amplitude (real)", -1.0, 1.0, 1/np.sqrt(2), step=0.01, key="p1_amp")
        path1_phase = st.slider("Path 1 Phase (radians)", 0.0, 2*np.pi, np.pi/2, step=0.01, key="p1_phase")
    
        # Calculate combined amplitude
        final_amp = (path0_amplitude * np.exp(1j * path0_phase)) + (path1_amplitude * np.exp(1j * path1_phase))
        final_prob = np.abs(final_amp)**2
    
        st.metric("Final Intensity (Probability)", f"{final_prob:.3f}")
    
    with col2:
        st.subheader("Visualization")
        fig_int, ax_int = plt.subplots(figsize=(6, 4))
        ax_int.set_title("Interference Pattern")
    
        # Visualize the two waves and their sum
        t = np.linspace(0, 2 * np.pi, 100)
        wave0 = path0_amplitude * np.cos(t - path0_phase)
        wave1 = path1_amplitude * np.cos(t - path1_phase)
        total_wave = wave0 + wave1
    
        ax_int.plot(t, wave0, label='Wave 0', color='blue', linestyle='--')
        ax_int.plot(t, wave1, label='Wave 1', color='orange', linestyle='--')
        ax_int.plot(t, total_wave, label='Combined Wave', color='red')
    
        ax_int.set_xlabel("Time/Position")
        ax_int.set_ylabel("Amplitude")
        ax_int.set_ylim(-2.5, 2.5)
        ax_int.legend()
        st.pyplot(fig_int)

    st.markdown("---")
    st.subheader("🎯 Challenges")
    
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        st.info("Challenge 1: Perfect Constructive Interference")
        if st.button("Check 1"):
            if abs(final_prob - (np.sqrt(2))**2) < 0.01:
                st.session_state.interference_challenges["max_constructive"] = True
                add_xp(20, "Achieved perfect constructive interference!")
            else:
                st.error("Hint: Make the two waves perfectly in phase.")

    with col_c2:
        st.info("Challenge 2: Perfect Destructive Interference")
        if st.button("Check 2"):
            if abs(final_prob) < 0.01:
                st.session_state.interference_challenges["max_destructive"] = True
                add_xp(20, "Achieved perfect destructive interference!")
            else:
                st.error("Hint: The waves need to cancel each other out.")

    with col_c3:
        st.info("Challenge 3: 50/50 Chance")
        if st.button("Check 3"):
            if abs(final_prob - 1) < 0.01:
                st.session_state.interference_challenges["half_prob"] = True
                add_xp(20, "Created a 50/50 probability!")
            else:
                st.error("Hint: The final wave's intensity should be 1.")

def stern_gerlach_lab():
    st.header("Stern–Gerlach Spin Measurement")
    st.markdown("This experiment demonstrates the **quantization of spin**. A beam of particles splits into discrete beams.")

    st.markdown("Imagine you're firing a beam of silver atoms, each with an intrinsic spin, through a non-uniform magnetic field.")
    st.write("Classically, you'd expect a continuous smear, but quantum mechanics predicts only two possible outcomes.")

    if st.button("▶️ Send a Particle", type="primary"):
        spin_state = np.random.choice(["Spin Up", "Spin Down"], p=[0.5, 0.5])
    
        st.info(f"Measured: **{spin_state}**")
    
        # Simple animation of particles splitting
        fig_sg, ax_sg = plt.subplots(figsize=(8, 4))
        ax_sg.set_title("Stern-Gerlach Experiment")
        ax_sg.set_xlim(0, 10)
        ax_sg.set_ylim(-3, 3)
        ax_sg.set_xticks([])
        ax_sg.set_yticks([])
        ax_sg.axvline(x=5, color='gray', linestyle='--')
    
        # Magnetic field representation
        ax_sg.text(5.1, 2, "North Pole (N)", ha='left', va='center')
        ax_sg.text(5.1, -2, "South Pole (S)", ha='left', va='center')
    
        # Path of the particle
        ax_sg.plot([0, 5], [0, 0], 'o-', color='black', label='Incoming Beam')
        if spin_state == "Spin Up":
            ax_sg.plot([5, 10], [0, 2], 'o-', color='blue')
            ax_sg.text(9, 2, "Up", ha='right', va='center', color='blue')
        else:
            ax_sg.plot([5, 10], [0, -2], 'o-', color='red')
            ax_sg.text(9, -2, "Down", ha='right', va='center', color='red')
    
        ax_sg.legend()
        st.pyplot(fig_sg)
    
        add_xp(10, "Simulated Stern-Gerlach experiment.")

def entanglement_workshop():
    st.header("Quantum Entanglement Lab")
    st.markdown("What Einstein called **'spooky action at a distance.'** When two qubits are entangled, their fates are linked.")
    st.markdown("Measuring one instantly affects the other, no matter the distance between them.")
    
    st.subheader("Bell State Generation")
    st.write("We will create a Bell state, $|Φ^+\\rangle = (|00\\rangle + |11\\rangle)/\\sqrt{2}$.")
    st.write("This state means the two qubits are perfectly correlated. If the first qubit is measured as $|0\\rangle$, the second will also be $|0\\rangle$. The same is true for $|1\\rangle$.")
    
    qc2 = QuantumCircuit(2)
    st.markdown("#### Quantum Circuit")
    qc2.h(0)
    qc2.cx(0, 1)
    st.code(qc2.draw('mpl')._repr_svg_())
    
    shots = st.slider("Number of measurements", 100, 2048, 1024)
    
    if st.button("🔬 Entangle & Measure", type="primary"):
        with st.spinner("Executing entanglement experiment..."):
            qc2.measure_all()
            simulator = Aer.get_backend('qasm_simulator')
            result = execute(qc2, simulator, shots=shots).result()
            counts = result.get_counts()
    
            st.success("Entanglement created: |Φ+> Bell state!")
    
            counts_df = pd.Series(counts)
            fig_counts = px.bar(
                x=counts_df.index,
                y=counts_df.values,
                labels={'x': 'Measurement Outcome', 'y': 'Counts'},
                title="Measurement Results for Entangled Qubits"
            )
            st.plotly_chart(fig_counts)
    
            if '01' not in counts and '10' not in counts:
                add_xp(30, "Confirmed spooky action! No uncorrelated outcomes.")
                if "Spooky Action Confirmed" not in st.session_state.achievements:
                    st.session_state.achievements.append("Spooky Action Confirmed")

def quantum_simulator():
    st.header("Quantum Simulator")
    st.markdown("Use a full quantum circuit simulator to run more complex experiments.")
    
    n_qubits = st.slider("Number of Qubits", 1, 4, 2, 1)
    st.info("The number of possible states doubles with each additional qubit, demonstrating the power of quantum scaling.")
    
    qc = QuantumCircuit(n_qubits)
    
    with st.expander("Add Gates"):
        gate_type = st.selectbox("Gate to add", ["Hadamard", "CNOT", "Pauli-X", "Pauli-Z", "Measurement"])
    
        if gate_type == "Hadamard":
            target_qubit = st.number_input("Target Qubit", 0, n_qubits - 1)
            if st.button("Add H Gate"):
                qc.h(target_qubit)
    
        elif gate_type == "CNOT":
            control_qubit = st.number_input("Control Qubit", 0, n_qubits - 1)
            target_qubit = st.number_input("Target Qubit", 0, n_qubits - 1)
            if control_qubit != target_qubit and st.button("Add CNOT Gate"):
                qc.cx(control_qubit, target_qubit)
    
        elif gate_type == "Pauli-X":
            target_qubit = st.number_input("Target Qubit", 0, n_qubits - 1)
            if st.button("Add X Gate"):
                qc.x(target_qubit)

        elif gate_type == "Pauli-Z":
            target_qubit = st.number_input("Target Qubit", 0, n_qubits - 1)
            if st.button("Add Z Gate"):
                qc.z(target_qubit)
    
        elif gate_type == "Measurement":
            if st.button("Add Measurement"):
                qc.measure_all()
    
    st.markdown("### Your Quantum Circuit")
    st.code(qc.draw('mpl')._repr_svg_())
    
    if st.button("Run Simulation", type="primary"):
        if not qc.data or not any(isinstance(op[0], (tuple, list)) for op in qc.data):
            st.error("Please add at least one gate to the circuit before running the simulation.")
            return

        with st.spinner("Simulating circuit..."):
            simulator = Aer.get_backend('qasm_simulator')
            result = execute(qc, simulator, shots=1024).result()
            counts = result.get_counts()
    
            st.success("Simulation complete!")
    
            counts_df = pd.Series(counts)
            fig_counts = px.bar(
                x=counts_df.index,
                y=counts_df.values,
                labels={'x': 'Measurement Outcome', 'y': 'Counts'},
                title="Simulation Results"
            )
            st.plotly_chart(fig_counts)
    
            add_xp(15, "Successfully ran a quantum simulation!")

def quizzes_and_challenges():
    st.header("Quantum Quizzes & Challenges")
    st.markdown("Test your knowledge and earn XP!")
    
    st.subheader("Quiz 1: Foundational Concepts")
    
    with st.expander("Question 1"):
        q1 = "What does the Hadamard gate do to a $|0\\rangle$ qubit?"
        a1 = st.radio(q1, ["Flips it to $|1\\rangle$", "Creates a superposition", "Measures it"])
        if st.button("Check Answer 1"):
            if a1 == "Creates a superposition":
                st.success("Correct! The Hadamard gate is a foundational tool for creating superposition.")
                add_xp(10, "Answered a quiz question correctly.")
            else:
                st.error("Not quite. The Hadamard gate rotates the state to create a superposition.")
                add_xp(3, "Attempted a quiz question.")
    
# Missing code for other quizzes would go here.

def future_quest_modules():
    st.header("🚀 Future Quest Modules")
    st.info("Stay tuned! More advanced modules are coming soon.")
    st.markdown("""
    - **Quantum Teleportation**: Learn how to transfer a quantum state from one qubit to another.
    - **Quantum Key Distribution (QKD)**: Explore the fundamentals of quantum cryptography.
    - **Grover's Algorithm**: Use a quantum algorithm to search an unsorted database faster than any classical computer.
    - **Shor's Algorithm**: Discover how a quantum computer could factor large numbers.
    """)

def resources_page():
    st.header("📚 Resources")
    st.markdown("### Learn More about Quantum Computing")
    st.info("Here are some great external resources to continue your quantum journey:")
    st.markdown("""
    - **[Qiskit Textbook](https://qiskit.org/textbook/preface.html)**: A comprehensive online textbook for learning quantum computing with Qiskit.
    - **[NIST Quantum Physics Portal](https://www.nist.gov/quantum)**: Learn about the latest research and applications from the National Institute of Standards and Technology.
    - **[The Feynman Lectures on Physics, Vol. III](https://www.feynmanlectures.caltech.edu/III_toc.html)**: A classic resource for a deeper dive into the foundations of quantum mechanics.
    """)

# --- Main App Entry Point ---
if __name__ == '__main__':
    main_app()
