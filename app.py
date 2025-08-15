import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import Tuple, Dict

# ------------------------------------------------------------
# MathCraft: Quantum Quest — An Interactive Intro to Quantum Mechanics
# Single-file Streamlit app (app.py)
# 
# Requirements (suggested):
#   streamlit>=1.33
#   numpy>=1.24
#   pandas>=2.0
#   plotly>=5.15 (optional; we avoid heavy plotting and use Streamlit widgets)
# ------------------------------------------------------------

st.set_page_config(
    page_title="MathCraft: Quantum Quest",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- Utility / Linear Algebra ----------------------------

def ket0() -> np.ndarray:
    return np.array([[1.0], [0.0]], dtype=complex)

def ket1() -> np.ndarray:
    return np.array([[0.0], [1.0]], dtype=complex)

def normalize(state: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(state)
    return state / norm if norm != 0 else state

# Basic single-qubit gates
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)        # Pauli-X (NOT)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)     # Pauli-Y
Z = np.array([[1, 0], [0, -1]], dtype=complex)       # Pauli-Z
H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)  # Hadamard

# Phase gate factory

def phase(theta: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)

# Rotation gates around X, Y, Z

def Rx(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                     [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)

def Ry(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)

def Rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

# Apply a gate to a state

def apply(gate: np.ndarray, state: np.ndarray) -> np.ndarray:
    return gate @ state

# Measure in computational basis and return probabilities and a sample

def measure_probs(state: np.ndarray) -> Dict[str, float]:
    a = state[0,0]
    b = state[1,0]
    p0 = float(np.real(a*np.conj(a)))
    p1 = float(np.real(b*np.conj(b)))
    return {"0": p0, "1": p1}

def sample_measure(state: np.ndarray) -> str:
    probs = measure_probs(state)
    return np.random.choice(["0","1"], p=[probs["0"], probs["1"]])

# Convert state to Bloch sphere coordinates (x,y,z)

def bloch_coords(state: np.ndarray) -> Tuple[float, float, float]:
    s = normalize(state)
    a = s[0,0]
    b = s[1,0]
    # Expectation values of Pauli matrices
    x = 2 * np.real(np.conj(a)*b)
    y = 2 * np.imag(np.conj(a)*b)
    z = float(np.real(np.conj(a)*a - np.conj(b)*b))
    return (float(x), float(y), float(z))

# Format complex number nicely

def cfmt(z: complex, precision: int = 3) -> str:
    return f"{z.real:.{precision}f} + {z.imag:.{precision}f}i"

# Pretty print a 2x2 matrix

def mat_to_df(M: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([[M[0,0], M[0,1]], [M[1,0], M[1,1]]], columns=["⟨0|", "⟨1|"], index=["|0⟩", "|1⟩"])

# ----------------------- App Header ----------------------------

st.title("MathCraft: Quantum Quest 🧪⚡")
st.caption("A hands-on, story-driven intro to qubits, superposition, and measurement — Built by Xavier Honablue M.Ed for CognitiveCloud.ai")

with st.sidebar:
    st.header("Navigate")
    page = st.radio(
        "Choose a module:",
        [
            "Story Mode",
            "Superposition Lab",
            "Quantum Gates Workshop",
            "Bloch Explorer",
            "Interference Sandbox",
            "Stern–Gerlach (Spin-1/2)",
            "Quizzes & Challenges",
            "Teacher Toolkit",
            "Resources"
        ]
    )
    st.divider()
    st.write("Progress")
    if "score" not in st.session_state:
        st.session_state.score = 0
    st.metric(label="XP", value=st.session_state.score)

# ----------------------- Story Mode ----------------------------

if page == "Story Mode":
    left, right = st.columns([3,2])
    with left:
        st.subheader("Episode 1: The Qubit in the Attic")
        st.write(
            """
            You find a dusty device labeled **Q-BOX**. A note reads:
            
            > *"Inside this box lives a qubit. It is both |0⟩ and |1⟩ until you look. 
            > Turn the dials to prepare its state, then measure wisely."*
            
            Your mission: prepare specific target states and measure outcomes to complete tasks.
            """
        )
        st.markdown("**Target State (Level 1):** the Hadamard state (|0⟩ + |1⟩)/√2")
        st.write("Use the dials to set rotation angles. Then measure 20 times and compare frequencies against the target (≈50% 0, 50% 1).")

        theta = st.slider("Rotate around Y (θ) to set superposition", 0.0, float(math.pi), value=float(math.pi/2), step=0.01)
        phi = st.slider("Phase twist around Z (φ)", 0.0, float(2*math.pi), value=0.0, step=0.01)

        # Start from |0>
        state = apply(Ry(theta), ket0())
        state = apply(Rz(phi), state)
        probs = measure_probs(state)

        st.write("**Current State |ψ⟩:**")
        st.code(f"|ψ⟩ = {cfmt(state[0,0])}|0⟩ + {cfmt(state[1,0])}|1⟩")
        st.write({"P(0)": probs['0'], "P(1)": probs['1']})

        trials = st.number_input("Number of measurements", min_value=10, max_value=500, value=50, step=10)
        if st.button("Run measurements"):
            results = [sample_measure(state) for _ in range(int(trials))]
            p0 = results.count("0") / len(results)
            p1 = 1 - p0
            st.success(f"Frequencies: 0 → {p0:.2f}, 1 → {p1:.2f}")
            # award XP if nearly balanced
            if abs(p0 - 0.5) < 0.08:
                st.balloons()
                st.session_state.score += 10
                st.info("Nice! Your distribution is close to the Hadamard target. +10 XP")

    with right:
        st.subheader("What’s Going On?")
        st.write(
            """
            A **qubit** is a unit vector in a 2D complex space. Rotations Ry(θ) and Rz(φ) move the state on the **Bloch sphere**.
            Measurement projects |ψ⟩ onto |0⟩ or |1⟩ with probabilities |α|² and |β|².
            """
        )
        st.markdown("**Tip:** Hadamard H maps |0⟩ → (|0⟩+|1⟩)/√2 and |1⟩ → (|0⟩−|1⟩)/√2.")

# ----------------------- Superposition Lab ----------------------------

elif page == "Superposition Lab":
    st.subheader("Build and measure superpositions")
    st.write("Use amplitudes α and β (subject to |α|²+|β|²=1). We’ll normalize automatically.")

    a_real = st.slider("Re(α)", -1.0, 1.0, 1.0, 0.01)
    a_imag = st.slider("Im(α)", -1.0, 1.0, 0.0, 0.01)
    b_real = st.slider("Re(β)", -1.0, 1.0, 0.0, 0.01)
    b_imag = st.slider("Im(β)", -1.0, 1.0, 0.0, 0.01)

    psi = np.array([[a_real + 1j*a_imag], [b_real + 1j*b_imag]], dtype=complex)
    psi = normalize(psi)
    st.code(f"|ψ⟩ = {cfmt(psi[0,0])}|0⟩ + {cfmt(psi[1,0])}|1⟩")

    probs = measure_probs(psi)
    st.write({"P(0)": probs['0'], "P(1)": probs['1']})

    x, y, z = bloch_coords(psi)
    st.write("**Bloch coordinates (x, y, z):**", (round(x,3), round(y,3), round(z,3)))

    nshots = st.number_input("Shots", 10, 1000, 100, 10)
    if st.button("Measure now"):
        results = [sample_measure(psi) for _ in range(int(nshots))]
        df = pd.Series(results).value_counts(normalize=True).rename_axis('outcome').reset_index(name='frequency')
        st.bar_chart(df.set_index('outcome'))
        st.session_state.score += 5
        st.info("Experiment complete! +5 XP")

# ----------------------- Quantum Gates Workshop ----------------------------

elif page == "Quantum Gates Workshop":
    st.subheader("Compose gates and watch the state evolve")

    gate_names = {
        "I (Identity)": I,
        "X (Pauli-X)": X,
        "Y (Pauli-Y)": Y,
        "Z (Pauli-Z)": Z,
        "H (Hadamard)": H,
    }

    st.write("Start state:")
    start_state = st.selectbox("Choose initial ket", ["|0⟩", "|1⟩"], index=0)
    state = ket0() if start_state == "|0⟩" else ket1()

    st.write("Add parameterized rotations and phase:")
    rx = st.slider("Rx(θ)", 0.0, float(2*math.pi), 0.0, 0.01)
    ry = st.slider("Ry(θ)", 0.0, float(2*math.pi), 0.0, 0.01)
    rz = st.slider("Rz(θ)", 0.0, float(2*math.pi), 0.0, 0.01)
    ph = st.slider("Phase φ", 0.0, float(2*math.pi), 0.0, 0.01)

    st.write("Pick a fixed gate to apply:")
    gate_choice = st.selectbox("Gate", list(gate_names.keys()), index=4)

    # Compose circuit
    circuit = phase(ph) @ Rz(rz) @ Ry(ry) @ Rx(rx) @ gate_names[gate_choice]
    out = apply(circuit, state)
    out = normalize(out)

    st.write("**Circuit matrix:**")
    st.dataframe(mat_to_df(circuit))

    st.write("**Output state:**")
    st.code(f"|ψ_out⟩ = {cfmt(out[0,0])}|0⟩ + {cfmt(out[1,0])}|1⟩")

    st.write("**Measurement probabilities:**", measure_probs(out))

    if st.button("Measure once"):
        outcome = sample_measure(out)
        st.success(f"You observed: {outcome}")

# ----------------------- Bloch Explorer ----------------------------

elif page == "Bloch Explorer":
    st.subheader("Visual intuition via Bloch coordinates")
    st.write("Adjust θ (from |0⟩ at θ=0 to |1⟩ at θ=π) and φ (relative phase).")

    theta = st.slider("Polar angle θ", 0.0, float(math.pi), float(math.pi/2), 0.001)
    phi = st.slider("Azimuthal angle φ", 0.0, float(2*math.pi), 0.0, 0.001)

    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    a = math.cos(theta/2)
    b = math.sin(theta/2) * np.exp(1j*phi)
    psi = np.array([[a],[b]], dtype=complex)

    x, y, z = bloch_coords(psi)
    st.code(f"|ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩\n(x,y,z)=({x:.3f},{y:.3f},{z:.3f})")
    st.write("**Probabilities:**", measure_probs(psi))

# ----------------------- Interference Sandbox ----------------------------

elif page == "Interference Sandbox":
    st.subheader("Toy double-slit interference with phases")
    st.write(
        """
        Two paths contribute complex amplitudes that add and then are squared to get probability.
        Adjust path phases and see constructive/destructive patterns.
        """
    )

    amp = st.slider("Base amplitude per path", 0.0, 1.0, 0.5, 0.01)
    phi1 = st.slider("Path 1 phase φ₁", 0.0, float(2*math.pi), 0.0, 0.01)
    phi2 = st.slider("Path 2 phase φ₂", 0.0, float(2*math.pi), float(math.pi), 0.01)

    a1 = amp * np.exp(1j*phi1)
    a2 = amp * np.exp(1j*phi2)
    total_amp = a1 + a2
    prob = float(np.real(total_amp * np.conj(total_amp)))

    st.write({"Amplitude a₁": cfmt(a1), "Amplitude a₂": cfmt(a2), "Total": cfmt(total_amp)})
    st.info(f"Resulting probability = |a₁+a₂|² = {prob:.3f}")

# ----------------------- Stern–Gerlach ----------------------------

elif page == "Stern–Gerlach (Spin-1/2)":
    st.subheader("Spin measurements along different axes")
    st.write(
        """
        Prepare a spin-1/2 state and measure along Z or X. 
        For example, start in |+⟩ = H|0⟩ and see 50/50 if measuring Z, or 100% + if measuring X.
        """
    )

    prep = st.selectbox("Preparation:", ["|0⟩ (spin up Z)", "|1⟩ (spin down Z)", "|+⟩ = H|0⟩", "|−⟩ = H|1⟩"], index=2)
    if prep == "|0⟩ (spin up Z)":
        psi = ket0()
    elif prep == "|1⟩ (spin down Z)":
        psi = ket1()
    elif prep == "|+⟩ = H|0⟩":
        psi = apply(H, ket0())
    else:
        psi = apply(H, ket1())

    axis = st.radio("Measure along:", ["Z", "X"])  # Y could be added similarly

    if axis == "Z":
        probs = measure_probs(psi)
        st.write("P(0) = ", probs['0'], ", P(1) = ", probs['1'])
    else:  # X basis: project onto |+⟩, |−⟩
        plus = apply(H, ket0())
        minus = apply(H, ket1())
        p_plus = float(np.real((np.conj(plus).T @ psi @ np.conj(psi).T @ plus).item()))
        # Simpler: probabilities via overlaps |⟨±|ψ⟩|²
        p_plus = float(np.abs((np.conj(plus).T @ psi).item())**2)
        p_minus = float(np.abs((np.conj(minus).T @ psi).item())**2)
        st.write({"P(+)": p_plus, "P(−)": p_minus})

    shots = st.number_input("Measurements", 10, 500, 100, 10)
    if st.button("Run Stern–Gerlach"):
        outcomes = []
        if axis == "Z":
            for _ in range(int(shots)):
                outcomes.append(sample_measure(psi))
            df = pd.Series(outcomes).value_counts(normalize=True).rename_axis('outcome').reset_index(name='frequency')
            st.bar_chart(df.set_index('outcome'))
        else:
            # Sample in X basis using overlaps
            plus = apply(H, ket0())
            p_plus = float(np.abs((np.conj(plus).T @ psi).item())**2)
            outcomes = np.random.choice(["+","−"], size=int(shots), p=[p_plus, 1-p_plus])
            df = pd.Series(outcomes).value_counts(normalize=True).rename_axis('outcome').reset_index(name='frequency')
            st.bar_chart(df.set_index('outcome'))
        st.session_state.score += 8
        st.info("Experiment complete! +8 XP")

# ----------------------- Quizzes & Challenges ----------------------------

elif page == "Quizzes & Challenges":
    st.subheader("Checkpoint Quiz")

    q1 = st.radio("1) What does the Hadamard gate do to |0⟩?", [
        "Leaves it unchanged",
        "Maps it to |1⟩",
        "Creates (|0⟩+|1⟩)/√2",
        "Applies a global phase only"
    ])

    q2 = st.radio("2) In |ψ⟩ = α|0⟩ + β|1⟩, what is P(1)?", [
        "|α|²",
        "|β|²",
        "αβ",
        "Re(β)"
    ])

    q3 = st.radio("3) Interference arises because…", [
        "Probabilities add directly",
        "Amplitudes (complex) add before squaring",
        "Measurements never disturb states",
        "Phases do not matter"
    ])

    q4 = st.radio("4) Which unitary rotates around the Z-axis by φ?", [
        "Rx(φ)", "Ry(φ)", "Rz(φ)", "H"
    ])

    if st.button("Submit Quiz"):
        score = 0
        score += 1 if q1 == "Creates (|0⟩+|1⟩)/√2" else 0
        score += 1 if q2 == "|β|²" else 0
        score += 1 if q3 == "Amplitudes (complex) add before squaring" else 0
        score += 1 if q4 == "Rz(φ)" else 0
        st.success(f"You scored {score}/4")
        st.session_state.score += 5 * score
        if score == 4:
            st.balloons()
            st.info("Perfect! Quantum Apprentice badge unlocked. +20 XP")

    st.divider()
    st.subheader("Challenges")
    st.markdown("1. **Equal Superposition:** Using rotations only, prepare ≈( |0⟩ + i|1⟩ )/√2.")
    st.markdown("2. **Phase Cancellation:** In Interference Sandbox, find φ₁, φ₂ for near-zero probability.")
    st.markdown("3. **Axis Flip:** In Stern–Gerlach, prepare a state that gives ~100% '+' when measuring X.")

# ----------------------- Teacher Toolkit ----------------------------

elif page == "Teacher Toolkit":
    st.subheader("Lesson Flow & Standards Alignment (HS/Intro UG)")

    st.markdown(
        """
        **Suggested 50–60 min lesson:**
        1. Hook (5 min): Story Mode intro, predict outcomes.
        2. Mini-lesson (10 min): Qubit, amplitudes, Bloch sphere, measurement.
        3. Guided practice (20 min): Superposition Lab + Gates Workshop.
        4. Experiments (10 min): Interference + Stern–Gerlach.
        5. Exit ticket (5 min): Quizzes & two challenges.

        **Objectives:** Students will
        - Describe a qubit as a normalized vector in C².
        - Compute probabilities from amplitudes (|α|², |β|²).
        - Explain how phase affects interference.
        - Use unitary gates to transform states.

        **Assessment:** Quiz, measurements logs, and challenge write-ups.

        **Differentiation:** Offer preset states (§Stern–Gerlach) and scaffolded sliders; extension challenges include composing arbitrary rotations to hit target Bloch coordinates.
        """
    )

    st.markdown("**Classroom Tips**")
    st.markdown("- Pair students for prediction–test cycles.\n- Emphasize distinction between amplitude vs probability.\n- Keep an eye on normalization; this app does it for them but talk through why.")

# ----------------------- Resources ----------------------------

elif page == "Resources":
    st.subheader("Further Study")
    st.markdown(
        """
        - *Quantum Computing for the Very Curious* (Michel Nielsen) — gentle, interactive notes.
        - *Qiskit Textbook* — comprehensive intro with circuits and code.
        - *Quantum Country* — spaced-repetition essays on quantum computing.
        - *Nielsen & Chuang* — the classic textbook for deeper study.
        - *Introduction to Quantum Mechanics* (Griffiths) — foundational QM.

        **Key Terms:** qubit, amplitude, superposition, phase, measurement, Bloch sphere, unitary, interference.
        """
    )

# ----------------------- Footer ----------------------------

st.markdown("---")
st.markdown("<div style='text-align:center'>© 2025 CognitiveCloud.ai — Built by Xavier Honablue M.Ed</div>", unsafe_allow_html=True)
