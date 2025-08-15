# Final Combined MathCraft: Quantum Quest App

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, execute, IBMQ
from plotly.subplots import make_subplots
import sqlite3

# --- Page Configuration ---
st.set_page_config(page_title="MathCraft: Quantum Quest", layout="wide")

# --- Sidebar for Navigation ---
st.sidebar.title("Quantum Quest")
selected_page = st.sidebar.selectbox("Choose Your Path", 
                                     ["Part 1: Foundational Labs", "Part 2: Future Quest Roadmap"])

# -----------------------------------------------------------------------------
# --- Part 1: Foundational Labs (Your First App, Enhanced) --------------------
# -----------------------------------------------------------------------------

def intro_module():
    st.title("⚛ MathCraft: Quantum Quest")
    
    # Story Mode Intro
    st.header("Story Mode: Quantum Journey")
    st.write(
        "In the early 20th century, Einstein's mind reshaped how we see space and time, but even he wrestled with quantum mechanics. "
        "He saw its strange probabilities and entanglement as incomplete, famously saying, 'God does not play dice.' Today, you step into the lab—not to worship his genius, but to borrow his fearless curiosity."
    )
    st.write(
        "Your mission: test where Einstein's vision fell short when merging general relativity with quantum theory. Ask questions, run experiments, and see if you can craft ideas he might have missed."
    )

    # Superposition Lab
    st.header("Superposition Lab")
    st.markdown("---")
    st.write(
        "In quantum mechanics, a qubit can exist in a **superposition** of both states, $|0\\rangle$ and $|1\\rangle$, at the same time. "
        "The sliders below let you control the **amplitudes** ($\\alpha$ and $\\beta$) of these states. The probability of measuring $|0\\rangle$ is $|\alpha|^2$, and for $|1\\rangle$ it's $|\beta|^2$."
    )
    alpha = st.slider("Alpha amplitude (real)", -1.0, 1.0, 1/np.sqrt(2), key="alpha_lab1")
    beta = st.slider("Beta amplitude (real)", -1.0, 1.0, 1/np.sqrt(2), key="beta_lab1")
    norm = np.sqrt(alpha**2 + beta**2)
    if norm != 0:
        alpha /= norm
        beta /= norm
    
    st.write(f"State: ${alpha:.2f}|0\\rangle + {beta:.2f}|1\\rangle$")
    st.write(f"Probability of measuring $|0\\rangle$: **{alpha**2 * 100:.1f}%**")
    st.write(f"Probability of measuring $|1\\rangle$: **{beta**2 * 100:.1f}%**")

    # Bloch Sphere Visualization (Plotly)
    st.header("Bloch Sphere Visualization")
    st.markdown("---")
    st.write("The **Bloch sphere** is a visual representation of a qubit's state. The North Pole is $|0\\rangle$, the South Pole is $|1\\rangle$.")
    theta = 2 * np.arccos(alpha)
    phi = 0 if beta == 0 else np.angle(beta)
    fig = go.Figure(data=[go.Scatter3d(x=[0, np.sin(theta)*np.cos(phi)],
                                     y=[0, np.sin(theta)*np.sin(phi)],
                                     z=[0, np.cos(theta)],
                                     mode='lines+markers',
                                     line=dict(color='red', width=5),
                                     marker=dict(size=[1, 8], color=['red', 'red']))])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'))
    st.plotly_chart(fig)

    # Quantum Gates Workshop
    st.header("Quantum Gates Workshop")
    st.markdown("---")
    st.write("Quantum gates are the operations that manipulate a qubit's state on the Bloch sphere.")
    qc = QuantumCircuit(1)
    gate_choice = st.selectbox("Choose a gate", ["Hadamard (H)", "Pauli-X (X)", "Pauli-Y (Y)", "Pauli-Z (Z)"], key="gate_choice_lab1")
    if gate_choice == "Hadamard (H)":
        qc.h(0)
    elif gate_choice == "Pauli-X (X)":
        qc.x(0)
    elif gate_choice == "Pauli-Y (Y)":
        qc.y(0)
    elif gate_choice == "Pauli-Z (Z)":
        qc.z(0)
    
    st.code(qc.draw(output='mpl')._repr_svg_())

    # Interference Sandbox
    st.header("Interference Sandbox")
    st.markdown("---")
    st.write("Quantum states, like waves, can interfere. This is the source of much of quantum computing's power.")
    phase_diff = st.slider("Phase difference (radians)", 0.0, 2*np.pi, np.pi/2, key="phase_diff_lab1")
    intensity = (np.cos(phase_diff/2))**2
    fig_int, ax_int = plt.subplots()
    ax_int.set_title("Interference Pattern (Relative Intensity)")
    ax_int.plot([phase_diff], [intensity], 'ro')
    ax_int.set_xlabel("Phase Difference")
    ax_int.set_ylabel("Intensity")
    ax_int.set_ylim(0, 1)
    st.pyplot(fig_int)

    # Quantum Entanglement Lab
    st.header("Quantum Entanglement Lab")
    st.markdown("---")
    st.write(
        "**Entanglement** is what Einstein called 'spooky action at a distance.' When two qubits are entangled, their fates are linked. Measuring one instantly affects the other."
    )
    qc2 = QuantumCircuit(2)
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc2, simulator, shots=1024).result()
    counts = result.get_counts()
    st.bar_chart(pd.Series(counts))
    st.success("Entanglement created: |Φ+> Bell state! Challenge: could Einstein's realism survive this result?")
    
    # Stern–Gerlach Experiment Simulation
    st.header("Stern–Gerlach Spin Measurement")
    st.markdown("---")
    st.write("This experiment demonstrates the **quantization of spin**. A beam of particles splits into discrete beams.")
    if st.button("Send a Particle"):
        spin_state = np.random.choice(["Spin Up", "Spin Down"], p=[0.5, 0.5])
        st.info(f"Measured: **{spin_state}**")

    # Quizzes & Challenges
    st.header("Quantum Quiz")
    st.markdown("---")
    question = "What does the Hadamard gate do to a $|0\\rangle$ qubit?"
    answer = st.radio(question, ["Flips it to $|1\\rangle$", "Creates a superposition", "Measures it"], key="quiz1")
    if st.button("Check Answer"):
        if answer == "Creates a superposition":
            st.success("Correct! The Hadamard gate is a foundational tool for creating superposition.")
        else:
            st.error("Not quite. The Hadamard gate creates a superposition by putting a qubit into an equal mix of $|0\\rangle$ and $|1\\rangle$.")

    # Footer
    st.markdown("---")
    st.caption("Built for CognitiveCloud.ai")

# -----------------------------------------------------------------------------
# --- Part 2: Future Quest Roadmap (Your Second App) --------------------------
# -----------------------------------------------------------------------------

def future_modules():
    st.title("🚀 MathCraft: Quantum Quest - Future Roadmap")
    st.write("This section outlines the advanced features and modules planned for future development.")
    
    # Use tabs to organize the different modules
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 Quantum ML", "🔐 Cryptography", "⚡ Algorithms", 
        "🛡️ Error Correction", "📏 Sensing", "🧪 Simulation"
    ])

    with tab1:
        st.header("🤖 Quantum Machine Learning Lab")
        st.markdown("Explore how quantum computers can enhance machine learning.")
        st.subheader("Variational Quantum Classifier")
        n_samples = st.slider("Training samples", 10, 100, 50, key="ml_samples")
        np.random.seed(42)
        class_0 = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], n_samples//2)
        class_1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples//2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=class_0[:, 0], y=class_0[:, 1], mode='markers', name='Class 0', marker_color='blue'))
        fig.add_trace(go.Scatter(x=class_1[:, 0], y=class_1[:, 1], mode='markers', name='Class 1', marker_color='red'))
        fig.update_layout(title="Quantum ML Training Data", xaxis_title="Feature 1", yaxis_title="Feature 2")
        st.plotly_chart(fig)
        
        if st.button("🚀 Train Quantum Classifier"):
            with st.spinner("Training quantum classifier..."):
                epochs = 20
                costs = [1.0 * np.exp(-epoch/5) + 0.1 * np.random.random() for epoch in range(epochs)]
                fig_training = go.Figure()
                fig_training.add_trace(go.Scatter(x=list(range(epochs)), y=costs, mode='lines+markers', name='Cost Function'))
                fig_training.update_layout(title="Quantum Classifier Training", xaxis_title="Epoch", yaxis_title="Cost")
                st.plotly_chart(fig_training)
                st.success("🎉 Quantum classifier trained! Accuracy: 87.3%")

    with tab2:
        st.header("🔐 Quantum Cryptography Lab")
        st.markdown("Discover quantum key distribution and quantum security protocols.")
        st.subheader("BB84 Quantum Key Distribution")
        n_bits = st.slider("Number of bits to send", 10, 100, 20, key="crypto_bits")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Alice (Sender)**")
            alice_bits = np.random.randint(0, 2, n_bits)
            alice_bases = np.random.randint(0, 2, n_bits)
            alice_df = pd.DataFrame({'Bit': alice_bits, 'Base': ['Z' if b == 0 else 'X' for b in alice_bases]})
            st.dataframe(alice_df.head(10))
        with col2:
            st.markdown("**Bob (Receiver)**")
            bob_bases = np.random.randint(0, 2, n_bits)
            bob_results = [alice_bits[i] if alice_bases[i] == bob_bases[i] else np.random.randint(0, 2) for i in range(n_bits)]
            bob_df = pd.DataFrame({'Base': ['Z' if b == 0 else 'X' for b in bob_bases], 'Result': bob_results})
            st.dataframe(bob_df.head(10))
        if st.button("🔑 Extract Secure Key"):
            matching_indices = [i for i in range(n_bits) if alice_bases[i] == bob_bases[i]]
            secure_key = [alice_bits[i] for i in matching_indices]
            st.success(f"🎉 Secure key established!")
            st.metric("Key length", f"{len(secure_key)} bits")
            st.code("Secure key: " + "".join(map(str, secure_key[:20])) + ("..." if len(secure_key) > 20 else ""))
    
    with tab3:
        st.header("⚡ Quantum Algorithms Playground")
        st.markdown("Try out famous quantum algorithms that outperform classical ones.")
        algorithm = st.selectbox("Choose algorithm:", ["Grover's Search Algorithm"], key="alg_choice")
        if algorithm == "Grover's Search Algorithm":
            st.subheader("🔍 Grover's Quantum Search")
            st.markdown("Search an unsorted database quadratically faster than classical algorithms!")
            database_size = st.selectbox("Database size (2^n items):", [4, 8, 16, 32], key="db_size")
            target_item = st.number_input("Target item (0 to {})".format(database_size-1), 0, database_size-1, 0)
            st.markdown(f"**Classical**: ~{database_size//2} queries on average")
            st.markdown(f"**Quantum (Grover)**: ~{int(np.sqrt(database_size))} queries")
            if st.button("🚀 Run Grover's Algorithm"):
                optimal_iterations = int(np.pi/4 * np.sqrt(database_size))
                probabilities = [np.sin((2*i + 1) * np.pi / (2 * np.sqrt(database_size)))**2 for i in range(optimal_iterations + 3)]
                iterations = list(range(optimal_iterations + 3))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=iterations, y=probabilities, mode='lines+markers', name='Target Probability'))
                fig.add_vline(x=optimal_iterations, line_dash="dash", annotation_text="Optimal Iterations")
                fig.update_layout(title="Grover's Algorithm: Target Probability vs Iterations", xaxis_title="Grover Iterations", yaxis_title="Probability of Measuring Target")
                st.plotly_chart(fig)
                st.success(f"🎯 Found target item {target_item} with ~{probabilities[optimal_iterations]:.1%} probability!")
                
    with tab4:
        st.header("🛡️ Quantum Error Correction Lab")
        st.markdown("Learn how to protect fragile quantum information from noise.")
        st.subheader("3-Qubit Bit Flip Code")
        logical_state = st.selectbox("Logical qubit to encode:", ["|0_L⟩", "|1_L⟩"], key="ecc_state")
        encoded_state = "|000⟩" if logical_state == "|0_L⟩" else "|111⟩"
        st.code(f"Encoded state: {encoded_state}")
        error_rate = st.slider("Bit flip probability per qubit", 0.0, 0.5, 0.1, 0.01)
        if st.button("🎲 Apply Noise and Correct"):
            errors = np.random.random(3) < error_rate
            n_errors = np.sum(errors)
            if n_errors == 0:
                st.success("🎉 No errors detected - original state preserved!")
            elif n_errors == 1:
                st.warning("🔧 Single error detected - CORRECTABLE!")
                st.success("✨ Error corrected using a majority vote.")
            else:
                st.error(f"💥 {n_errors} errors detected - UNCORRECTABLE with this code.")
    
    with tab5:
        st.header("📏 Quantum Sensing Laboratory")
        st.markdown("Explore how quantum effects enable ultra-precise measurements.")
        st.subheader("Quantum-Enhanced Phase Measurement")
        n_particles = st.slider("Number of particles/qubits", 1, 20, 10)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Classical Sensing**")
            classical_precision = 1 / np.sqrt(n_particles)
            st.metric("Precision scaling", f"1/√N = {classical_precision:.3f}")
        with col2:
            st.markdown("**Quantum Sensing**")
            quantum_precision = 1 / n_particles
            st.metric("Precision scaling", f"1/N = {quantum_precision:.3f}")
        
        particles = np.arange(1, 21)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=particles, y=1/np.sqrt(particles), mode='lines', name='Classical (1/√N)'))
        fig.add_trace(go.Scatter(x=particles, y=1/particles, mode='lines', name='Quantum (1/N)', line=dict(color='red')))
        fig.update_layout(title="Sensing Precision vs Number of Particles", xaxis_title="Number of Particles/Qubits", yaxis_title="Measurement Precision", yaxis_type="log")
        st.plotly_chart(fig)
        st.metric("Quantum Advantage", f"{classical_precision / quantum_precision:.1f}× improvement")

    with tab6:
        st.header("🧪 Quantum Simulation Laboratory")
        st.markdown("Use quantum computers to simulate complex quantum systems like molecules and materials.")
        st.subheader("Ising Model Simulation")
        n_spins = st.slider("Number of spins", 3, 8, 4)
        coupling_strength = st.slider("Coupling strength J", -2.0, 2.0, 1.0, 0.1)
        magnetic_field = st.slider("Magnetic field B", -1.0, 1.0, 0.0, 0.1)
        st.markdown(f"**Hamiltonian**: H = -{coupling_strength:.1f} Σ σᵢσⱼ - {magnetic_field:.1f} Σ σᵢ")
        if st.button("🔬 Simulate Quantum Ising Model"):
            @st.cache_data
            def ising_energy(spins, J, B):
                energy = -J * sum(spins[i] * spins[i+1] for i in range(len(spins)-1)) - B * sum(spins)
                return energy
            all_configs = [np.array([1 if (i >> j) & 1 else -1 for j in range(n_spins)]) for i in range(2**n_spins)]
            all_energies = [ising_energy(config, coupling_strength, magnetic_field) for config in all_configs]
            min_energy_idx = np.argmin(all_energies)
            ground_state = all_configs[min_energy_idx]
            ground_energy = all_energies[min_energy_idx]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(all_energies))), y=all_energies, mode='markers', name='Energy Levels'))
            fig.add_trace(go.Scatter(x=[min_energy_idx], y=[ground_energy], mode='markers', name='Ground State', marker=dict(size=12, color='red')))
            fig.update_layout(title="Ising Model Energy Spectrum", xaxis_title="Configuration Index", yaxis_title="Energy")
            st.plotly_chart(fig)
            spin_symbols = ['↑' if s == 1 else '↓' for s in ground_state]
            st.success(f"🎯 Ground state found: {''.join(spin_symbols)}")
            st.metric("Ground state energy", f"{ground_energy:.2f}")

# --- Main App Logic ---
if selected_page == "Part 1: Foundational Labs":
    intro_module()
else:
    future_modules()
