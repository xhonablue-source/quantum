# Enhanced MathCraft: Quantum Quest App with Tutoring & Advanced Rewards

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
import random

# --- Page Configuration ---
st.set_page_config(
    page_title="MathCraft: Quantum Quest Enhanced",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced session state for the new features
if 'xp' not in st.session_state:
    st.session_state.xp = 0
if 'quantum_coins' not in st.session_state:
    st.session_state.quantum_coins = 0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'current_level' not in st.session_state:
    st.session_state.current_level = 1
if 'completed_modules' not in st.session_state:
    st.session_state.completed_modules = set()
if 'quiz_scores' not in st.session_state:
    st.session_state.quiz_scores = {}
if 'tutoring_progress' not in st.session_state:
    st.session_state.tutoring_progress = {}
if 'quantum_streak' not in st.session_state:
    st.session_state.quantum_streak = 0
if 'mastery_levels' not in st.session_state:
    st.session_state.mastery_levels = {
        'superposition': 0,
        'gates': 0,
        'entanglement': 0,
        'measurement': 0,
        'interference': 0
    }
if 'shop_inventory' not in st.session_state:
    st.session_state.shop_inventory = {
        'hint_booster': 0,
        'xp_multiplier': 0,
        'theme_dark': False,
        'theme_neon': False,
        'advanced_bloch': False
    }

# ----------------------- Core Quantum Functions ----------------------------

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

# ----------------------- Enhanced Reward System ----------------------------

def quantum_reward_calculation(base_xp: int, difficulty: str, streak_bonus: bool = False) -> Dict:
    """Enhanced quantum reward system with multiple reward types"""
    
    difficulty_multipliers = {
        'easy': 1.0,
        'medium': 1.5,
        'hard': 2.0,
        'expert': 3.0
    }
    
    multiplier = difficulty_multipliers.get(difficulty, 1.0)
    
    # Base rewards
    xp_reward = int(base_xp * multiplier)
    coin_reward = int(base_xp * multiplier * 0.5)
    
    # Streak bonus
    if streak_bonus and st.session_state.quantum_streak >= 3:
        streak_multiplier = 1 + (st.session_state.quantum_streak * 0.1)
        xp_reward = int(xp_reward * streak_multiplier)
        coin_reward = int(coin_reward * streak_multiplier)
    
    # Random quantum bonus (5% chance for double rewards)
    quantum_bonus = random.random() < 0.05
    if quantum_bonus:
        xp_reward *= 2
        coin_reward *= 2
    
    return {
        'xp': xp_reward,
        'coins': coin_reward,
        'quantum_bonus': quantum_bonus,
        'streak_bonus': streak_bonus and st.session_state.quantum_streak >= 3
    }

def add_xp_enhanced(base_points: int, reason: str = "", difficulty: str = 'easy', correct_answer: bool = True):
    """Enhanced XP system with quantum rewards"""
    
    if correct_answer:
        st.session_state.quantum_streak += 1
        rewards = quantum_reward_calculation(base_points, difficulty, streak_bonus=True)
    else:
        st.session_state.quantum_streak = 0
        rewards = {'xp': max(1, base_points // 3), 'coins': 0, 'quantum_bonus': False, 'streak_bonus': False}
    
    # Apply rewards
    st.session_state.xp += rewards['xp']
    st.session_state.quantum_coins += rewards['coins']
    
    # Display reward message
    reward_msg = f"+{rewards['xp']} XP"
    if rewards['coins'] > 0:
        reward_msg += f", +{rewards['coins']} ⚛️"
    
    if rewards['quantum_bonus']:
        reward_msg += " 🌟 QUANTUM BONUS!"
        st.balloons()
    
    if rewards['streak_bonus']:
        reward_msg += f" 🔥 Streak x{st.session_state.quantum_streak}!"
    
    if reason:
        reward_msg += f": {reason}"
    
    if correct_answer:
        st.success(reward_msg)
    else:
        st.info(reward_msg)
    
    # Level up check
    new_level = (st.session_state.xp // 100) + 1
    if new_level > st.session_state.current_level:
        st.session_state.current_level = new_level
        st.balloons()
        st.success(f"🎉 Level Up! You're now Level {new_level}!")
        st.session_state.quantum_coins += new_level * 10

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
    
    # Axes
    fig.add_trace(go.Scatter3d(x=[-1.2, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.2, 1.2], z=[0, 0], mode='lines', line=dict(color='black', width=2), showlegend=False))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.2, 1.2], mode='lines', line=dict(color='black', width=2), showlegend=False))
    
    # Labels
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

# ----------------------- Tutoring System ----------------------------

class QuantumTutor:
    """Comprehensive tutoring system for quantum concepts"""
    
    @staticmethod
    def get_concept_explanation(concept: str) -> Dict:
        explanations = {
            'superposition': {
                'title': '🌟 Quantum Superposition',
                'simple': "A qubit can be in multiple states at the same time! Unlike a classical bit that's either 0 or 1, a qubit can be both 0 AND 1 simultaneously.",
                'detailed': """
                **Mathematical Description:**
                A qubit in superposition is written as: |ψ⟩ = α|0⟩ + β|1⟩
                
                Where:
                - α and β are complex numbers (amplitudes)
                - |α|² gives the probability of measuring |0⟩
                - |β|² gives the probability of measuring |1⟩
                - |α|² + |β|² = 1 (normalization condition)
                
                **Physical Intuition:**
                Think of a spinning coin in the air - before it lands, it's neither heads nor tails, but both!
                """,
                'examples': [
                    "Equal superposition: |+⟩ = (|0⟩ + |1⟩)/√2 - 50% chance each",
                    "Unequal superposition: 0.6|0⟩ + 0.8|1⟩ - 36% chance |0⟩, 64% chance |1⟩"
                ],
                'common_mistakes': [
                    "❌ Thinking the qubit 'chooses' a state before measurement",
                    "❌ Confusing probability amplitudes with probabilities"
                ]
            },
            'gates': {
                'title': '⚙️ Quantum Gates',
                'simple': "Quantum gates are operations that transform qubit states. They're like functions that take a quantum state as input and produce a new quantum state as output.",
                'detailed': """
                **Key Properties:**
                - Quantum gates are reversible (unitary operations)
                - They preserve the total probability
                - Gates can create, manipulate, or destroy superposition
                
                **Common Gates:**
                - **X Gate**: Quantum NOT - flips |0⟩ ↔ |1⟩
                - **H Gate**: Hadamard - creates superposition from basis states
                - **Z Gate**: Phase flip - adds minus sign to |1⟩ state
                """,
                'examples': [
                    "H|0⟩ = (|0⟩ + |1⟩)/√2 - creates equal superposition",
                    "X|0⟩ = |1⟩ - classical bit flip"
                ],
                'common_mistakes': [
                    "❌ Thinking gates 'measure' the qubit",
                    "❌ Forgetting gates are matrix operations"
                ]
            },
            'measurement': {
                'title': '📏 Quantum Measurement',
                'simple': "Measurement collapses a superposition into a definite state. It's like asking 'What state is the qubit in?' and getting a definite answer.",
                'detailed': """
                **The Measurement Process:**
                1. Before: |ψ⟩ = α|0⟩ + β|1⟩ (superposition)
                2. Measure: Device asks "Is it |0⟩ or |1⟩?"
                3. After: Either |0⟩ (with probability |α|²) or |1⟩ (with probability |β|²)
                
                **Key Points:**
                - Measurement is probabilistic
                - It destroys superposition (irreversible)
                - Repeated measurements give the same result
                """,
                'examples': [
                    "Measuring |+⟩ gives |0⟩ or |1⟩ with 50% chance each",
                    "Measuring |0⟩ always gives |0⟩ (100% probability)"
                ],
                'common_mistakes': [
                    "❌ Thinking you can measure without disturbing the system",
                    "❌ Expecting deterministic outcomes from superposition states"
                ]
            },
            'entanglement': {
                'title': '🔗 Quantum Entanglement',
                'simple': "When qubits become entangled, they're connected in a spooky way - measuring one instantly affects the other!",
                'detailed': """
                **Bell States (Maximally Entangled):**
                - |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 - both qubits always agree
                - |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 - qubits always disagree
                
                **Properties:**
                - Cannot be written as a product of individual qubit states
                - Measuring one qubit instantly determines the other's state
                """,
                'examples': [
                    "In |Φ⁺⟩: If first qubit measures |0⟩, second is definitely |0⟩",
                    "In |Ψ⁺⟩: If first qubit measures |0⟩, second is definitely |1⟩"
                ],
                'common_mistakes': [
                    "❌ Thinking entanglement allows faster-than-light communication",
                    "❌ Confusing correlation with causation"
                ]
            }
        }
        return explanations.get(concept, {})
    
    @staticmethod
    def adaptive_hint(concept: str, difficulty_level: int) -> str:
        """Provides hints based on user's current understanding level"""
        hints = {
            'superposition': [
                "Think of a coin spinning in the air - it's both heads and tails until it lands!",
                "The key insight: probability amplitudes can be negative or complex numbers.",
                "Try adjusting the Hadamard gate to see how it creates equal superposition.",
                "Advanced: Consider how phases affect interference between amplitudes."
            ],
            'gates': [
                "Gates are like recipes - they tell you how to transform your quantum state.",
                "The X gate is just like a classical NOT gate, but for qubits.",
                "Hadamard gates are superposition creators - they're the most important gate to understand.",
                "Try combining different gates to see how they compose into new operations."
            ],
            'measurement': [
                "Measurement is like asking the qubit a yes/no question.",
                "The probabilities come from squaring the amplitude magnitudes.",
                "Once measured, the superposition is gone forever!",
                "Advanced: Consider how measurement bases affect the outcomes."
            ]
        }
        
        concept_hints = hints.get(concept, ["Keep exploring to learn more!"])
        return concept_hints[min(difficulty_level, len(concept_hints) - 1)]

def show_concept_tutor(concept: str):
    """Interactive tutoring interface for a specific concept"""
    
    tutor_data = QuantumTutor.get_concept_explanation(concept)
    if not tutor_data:
        return
    
    st.subheader(f"📚 {tutor_data['title']}")
    
    # Difficulty level selection
    difficulty = st.radio(
        "Choose explanation level:",
        ["🌱 Beginner", "🔬 Intermediate", "🎓 Advanced"],
        key=f"tutor_{concept}"
    )
    
    if difficulty == "🌱 Beginner":
        st.info(tutor_data['simple'])
        
        # Interactive example for beginners
        if concept == 'superposition':
            st.markdown("**Try it yourself:**")
            alpha = st.slider("Amplitude for |0⟩", 0.0, 1.0, 0.707, key=f"alpha_{concept}")
            beta = np.sqrt(1 - alpha**2)
            st.write(f"Your state: {alpha:.3f}|0⟩ + {beta:.3f}|1⟩")
            st.write(f"Probability of |0⟩: {alpha**2:.3f}")
            st.write(f"Probability of |1⟩: {beta**2:.3f}")
    
    elif difficulty == "🔬 Intermediate":
        st.markdown(tutor_data['detailed'])
        
        if 'examples' in tutor_data:
            st.markdown("**Examples:**")
            for example in tutor_data['examples']:
                st.markdown(f"• {example}")
    
    else:  # Advanced
        st.markdown(tutor_data['detailed'])
        
        if 'examples' in tutor_data:
            st.markdown("**Examples:**")
            for example in tutor_data['examples']:
                st.markdown(f"• {example}")
        
        if 'common_mistakes' in tutor_data:
            st.markdown("**⚠️ Common Mistakes to Avoid:**")
            for mistake in tutor_data['common_mistakes']:
                st.markdown(f"• {mistake}")
    
    # Adaptive hint system
    user_level = st.session_state.mastery_levels.get(concept, 0)
    hint = QuantumTutor.adaptive_hint(concept, user_level)
    st.success(f"💡 **Hint:** {hint}")

# ----------------------- Quiz System ----------------------------

def run_simple_quiz():
    """Simplified quiz system"""
    st.header("🧠 Quantum Quiz Center")
    
    # Quiz questions
    quiz_questions = [
        {
            'question': "What does the Hadamard gate do to a |0⟩ qubit?",
            'options': ["Flips it to |1⟩", "Creates an equal superposition", "Measures it", "Does nothing"],
            'correct': 1,
            'explanation': "The Hadamard gate creates an equal superposition: H|0⟩ = (|0⟩ + |1⟩)/√2",
            'difficulty': 'easy'
        },
        {
            'question': "In the state |ψ⟩ = 0.6|0⟩ + 0.8|1⟩, what's the probability of measuring |0⟩?",
            'options': ["0.6", "0.36", "0.8", "0.5"],
            'correct': 1,
            'explanation': "Probability = |amplitude|². So P(|0⟩) = |0.6|² = 0.36",
            'difficulty': 'medium'
        },
        {
            'question': "Which gate adds a phase of -1 to the |1⟩ state?",
            'options': ["X gate", "Y gate", "Z gate", "H gate"],
            'correct': 2,
            'explanation': "The Z gate does: Z|0⟩ = |0⟩ and Z|1⟩ = -|1⟩",
            'difficulty': 'medium'
        },
        {
            'question': "What happens to a qubit in superposition when measured?",
            'options': ["It stays in superposition", "It collapses to a definite state", "It becomes entangled", "It disappears"],
            'correct': 1,
            'explanation': "Measurement causes the superposition to collapse into one of the basis states.",
            'difficulty': 'easy'
        },
        {
            'question': "Which Bell state represents (|00⟩ + |11⟩)/√2?",
            'options': ["|Φ⁺⟩", "|Φ⁻⟩", "|Ψ⁺⟩", "|Ψ⁻⟩"],
            'correct': 0,
            'explanation': "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 is the classic Bell state where both qubits are correlated.",
            'difficulty': 'hard'
        }
    ]
    
    # Initialize quiz state
    if 'current_quiz_q' not in st.session_state:
        st.session_state.current_quiz_q = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_completed = False
    
    if not st.session_state.quiz_completed:
        current_q = st.session_state.current_quiz_q
        
        if current_q < len(quiz_questions):
            question = quiz_questions[current_q]
            
            # Progress
            progress = (current_q + 1) / len(quiz_questions)
            st.progress(progress)
            st.markdown(f"**Question {current_q + 1} of {len(quiz_questions)}**")
            
            # Question
            st.markdown(f"### {question['question']}")
            
            # Options
            answer = st.radio(
                "Choose your answer:",
                question['options'],
                key=f"quiz_q_{current_q}"
            )
            
            if st.button("Submit Answer", type="primary"):
                is_correct = question['options'].index(answer) == question['correct']
                
                if is_correct:
                    st.session_state.quiz_score += 1
                    st.success("✅ Correct!")
                    add_xp_enhanced(20, "Correct answer!", question['difficulty'], True)
                else:
                    st.error("❌ Incorrect")
                    correct_answer = question['options'][question['correct']]
                    st.info(f"Correct answer: {correct_answer}")
                    add_xp_enhanced(5, "Keep learning!", question['difficulty'], False)
                
                st.info(f"💡 {question['explanation']}")
                st.session_state.current_quiz_q += 1
                
                time.sleep(2)
                st.experimental_rerun()
        else:
            st.session_state.quiz_completed = True
            st.experimental_rerun()
    
    else:
        # Show results
        st.success("🎉 Quiz Completed!")
        
        score = st.session_state.quiz_score
        total = len(quiz_questions)
        percentage = (score / total) * 100
        
        st.metric("Final Score", f"{score}/{total} ({percentage:.1f}%)")
        
        if percentage >= 80:
            st.success("🌟 Excellent work!")
            if "Quiz Master" not in st.session_state.achievements:
                st.session_state.achievements.append("Quiz Master")
        elif percentage >= 60:
            st.info("👍 Good job!")
        else:
            st.warning("📚 Keep studying!")
        
        if st.button("🔄 Take Quiz Again"):
            st.session_state.current_quiz_q = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_completed = False
            st.experimental_rerun()

# ----------------------- Main Modules ----------------------------

def story_mode_enhanced():
    st.header("Episode 1: The Mysterious Q-Box Enhanced")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📖 The Enhanced Discovery
        
        In the dusty attic of an old physics laboratory, you discover a peculiar device labeled **Q-BOX 2.0**. 
        This isn't just any quantum device - it's equipped with an AI tutor and advanced quantum mechanics!
        
        A glowing holographic note appears:
        
        > *"Welcome, quantum explorer! This enhanced Q-Box will guide you through the mysteries of quantum mechanics.
        > Complete challenges, earn quantum coins, and unlock the deepest secrets of the quantum realm."*
        
        Your enhanced quantum journey begins now! 🚀
        """)
        
        st.markdown("### 🎯 Mission: Create the Hadamard State")
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
                    add_xp_enhanced(25, "Successfully created superposition state!", 'medium', True)
                    if "Hadamard Master" not in st.session_state.achievements:
                        st.session_state.achievements.append("Hadamard Master")
                        st.balloons()
                else:
                    add_xp_enhanced(8, "Good attempt! Try adjusting θ to π/2 for perfect superposition.", 'easy', False)
    
    with col2:
        st.markdown("### 🧠 Enhanced Quantum Concepts")
        
        # AI Tutor integration
        if st.button("🎓 Ask AI Tutor"):
            show_concept_tutor('superposition')
        
        st.markdown("""
        **Superposition**: A qubit can exist in a combination of |0⟩ and |1⟩ states simultaneously.
        
        **Measurement**: Collapses the superposition, giving either |0⟩ or |1⟩ with probabilities |α|² and |β|².
        
        **Phase**: The relative phase between states affects interference but not individual measurement probabilities.
        """)
        
        x, y, z = bloch_coords(state)
        fig = create_bloch_sphere(x, y, z, "Your Qubit State")
        st.plotly_chart(fig, use_container_width=True)

def ai_quantum_tutor():
    st.header("🎓 AI Quantum Tutor")
    st.markdown("Your personal AI assistant for mastering quantum mechanics!")
    
    # Concept selection
    st.subheader("📚 Choose a Concept to Learn")
    
    concepts = ['superposition', 'gates', 'measurement', 'entanglement']
    concept_names = ['Superposition', 'Quantum Gates', 'Measurement', 'Entanglement']
    
    selected_concept = st.selectbox(
        "What would you like to learn about?",
        concepts,
        format_func=lambda x: concept_names[concepts.index(x)]
    )
    
    # Show mastery level
    mastery = st.session_state.mastery_levels.get(selected_concept, 0)
    st.markdown(f"**Your current mastery level:** {'⭐' * mastery}{'☆' * (3 - mastery)}")
    
    # Interactive tutor
    show_concept_tutor(selected_concept)
    
    # Practice recommendations
    st.subheader("🎯 Recommended Practice")
    
    if mastery == 0:
        st.info("Start with the basic quizzes and simple lab exercises!")
    elif mastery == 1:
        st.info("Try intermediate challenges and explore the advanced lab features!")
    elif mastery == 2:
        st.info("You're almost there! Tackle the hardest challenges to achieve mastery!")
    else:
        st.success("🌟 You've mastered this concept! Help others or explore advanced topics!")

def superposition_lab_enhanced():
    st.header("🧪 Enhanced Superposition Laboratory")
    st.markdown("Craft custom quantum states with AI guidance and advanced features")
    
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
                
                add_xp_enhanced(12, "Superposition experiment completed!", 'easy', True)
    
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

def quantum_gates_workshop_enhanced():
    st.header("⚙️ Enhanced Quantum Gates Workshop")
    st.markdown("Master quantum gates with interactive tutorials")
    
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
            st.markdown("# ➡️")
        
        with col_final:
            st.markdown("**Final State:**")
            st.code(f"|ψ⟩ = {cfmt(final_state[0,0])}|0⟩ + {cfmt(final_state[1,0])}|1⟩")
            final_probs = measure_probs(final_state)
            st.write(f"P(0) = {final_probs['0']:.3f}")
            st.write(f"P(1) = {final_probs['1']:.3f}")
        
        if st.button("🎯 Simulate Circuit", type="primary"):
            add_xp_enhanced(15, "Simulated quantum circuit!", 'medium', True)
            st.success("Circuit simulation completed!")
    
    with col2:
        x, y, z = bloch_coords(final_state)
        fig = create_bloch_sphere(x, y, z, "Final State on Bloch Sphere")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🧠 Gate Reference")
        st.markdown("""
        - **X**: Bit flip (|0⟩ ↔ |1⟩)
        - **H**: Creates superposition
        - **Z**: Phase flip (adds minus to |1⟩)
        - **Rx/Ry/Rz**: Rotation around X/Y/Z axis
        - **Phase**: Adds phase to |1⟩ state
        """)

def quantum_challenges():
    st.header("🎮 Quantum Challenges")
    st.markdown("Test your skills with these quantum mechanics challenges!")
    
    st.subheader("State Creation Challenges")
    st.markdown("Create specific quantum states using the minimum number of operations!")
    
    challenges = [
        {
            "name": "The Perfect Balance",
            "description": "Create a state with exactly 30% probability of measuring |0⟩",
            "target_prob": 0.3,
            "difficulty": "medium",
            "reward": 25
        },
        {
            "name": "Quantum Coin Flip",
            "description": "Create perfect 50/50 superposition starting from |1⟩",
            "target_prob": 0.5,
            "difficulty": "easy",
            "reward": 15
        }
    ]
    
    challenge = st.selectbox("Choose a challenge:", challenges, format_func=lambda x: x['name'])
    
    st.markdown(f"**🎯 Challenge:** {challenge['name']}")
    st.markdown(f"**📝 Description:** {challenge['description']}")
    st.markdown(f"**⚡ Difficulty:** {challenge['difficulty'].title()}")
    st.markdown(f"**🏆 Reward:** {challenge['reward']} XP + coins")
    
    # Simple challenge interface
    st.subheader("🔧 Your Solution")
    
    operation = st.selectbox("Choose operation:", ["H", "X", "Y", "Z"])
    
    if st.button("🧪 Test Solution", type="primary"):
        # Simulate based on challenge
        if challenge['name'] == "Quantum Coin Flip" and operation == "H":
            st.success("🎉 Challenge completed!")
            add_xp_enhanced(challenge['reward'], f"Completed {challenge['name']}!", challenge['difficulty'], True)
        elif challenge['name'] == "The Perfect Balance":
            st.info("💡 Try using a combination of rotation gates to achieve the exact probability!")
        else:
            st.error("❌ Not quite right. Think about what each gate does!")

def quantum_shop():
    st.header("💎 Quantum Shop")
    st.markdown("Spend your quantum coins on useful upgrades and cosmetics!")
    
    st.markdown(f"**Your Quantum Coins:** ⚛️ {st.session_state.quantum_coins}")
    
    # Shop items
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💡 Hint Booster")
        st.markdown("Get extra hints in quizzes")
        st.markdown("**Price:** ⚛️ 50")
        st.markdown(f"**Owned:** {st.session_state.shop_inventory['hint_booster']}")
        
        if st.button("Buy Hint Booster"):
            if st.session_state.quantum_coins >= 50:
                st.session_state.quantum_coins -= 50
                st.session_state.shop_inventory['hint_booster'] += 1
                st.success("Purchased! You now have extra hints available.")
            else:
                st.error("Not enough quantum coins!")
    
    with col2:
        st.markdown("### ⚡ XP Multiplier")
        st.markdown("2x XP for next 10 activities")
        st.markdown("**Price:** ⚛️ 100")
        st.markdown(f"**Owned:** {st.session_state.shop_inventory['xp_multiplier']}")
        
        if st.button("Buy XP Multiplier"):
            if st.session_state.quantum_coins >= 100:
                st.session_state.quantum_coins -= 100
                st.session_state.shop_inventory['xp_multiplier'] += 1
                st.success("Purchased! Your next 10 activities will give 2x XP!")
            else:
                st.error("Not enough quantum coins!")
    
    with col3:
        st.markdown("### 🌙 Dark Theme")
        st.markdown("Sleek dark theme for night studying")
        st.markdown("**Price:** ⚛️ 150")
        
        if st.session_state.shop_inventory['theme_dark']:
            st.success("✅ Owned")
        else:
            if st.button("Buy Dark Theme"):
                if st.session_state.quantum_coins >= 150:
                    st.session_state.quantum_coins -= 150
                    st.session_state.shop_inventory['theme_dark'] = True
                    st.success("Theme purchased!")
                else:
                    st.error("Not enough quantum coins!")

def bloch_explorer():
    st.header("Bloch Explorer")
    st.markdown("Freely navigate the Bloch sphere and observe the state vector in real-time.")
    
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

def resources_page():
    st.header("📚 Quantum Learning Resources")
    st.markdown("### Continue Your Quantum Journey")
    
    st.subheader("Recommended Books")
    st.markdown("""
    - **Quantum Computing: An Applied Approach** by Hidary, J. D.
    - **Quantum Computation and Quantum Information** by Nielsen & Chuang
    - **Programming Quantum Computers** by Johnston, Harrigan & Gimeno-Segovia
    """)
    
    st.subheader("Online Courses")
    st.markdown("""
    - **IBM Qiskit Textbook** - Comprehensive online resource
    - **Microsoft Quantum Development Kit** - Learn Q# programming
    - **edX Quantum Mechanics Courses** - University-level content
    """)
    
    st.subheader("🎯 Study Tips")
    study_tips = [
        "Start with the mathematical foundations - linear algebra and complex numbers",
        "Practice with simulators before trying real quantum hardware",
        "Join quantum computing communities and forums for discussions",
        "Work through problems step-by-step, don't rush concepts"
    ]
    
    for i, tip in enumerate(study_tips, 1):
        st.markdown(f"{i}. {tip}")

def future_quest_modules():
    st.header("🚀 Future Quest Modules")
    st.info("Stay tuned! More advanced modules are coming soon.")
    
    upcoming_features = [
        "🌐 Quantum Teleportation Lab",
        "🔐 Quantum Cryptography Workshop", 
        "🔍 Grover's Search Algorithm",
        "🧮 Shor's Factoring Algorithm",
        "🎮 Quantum Games & Puzzles"
    ]
    
    for feature in upcoming_features:
        st.markdown(f"• {feature}")

# ----------------------- Main App Structure ----------------------------

def main_app():
    st.title("⚛️ MathCraft: Quantum Quest Enhanced")
    st.caption("A comprehensive, story-driven introduction to quantum mechanics with AI tutoring")

    # --- Enhanced Sidebar ---
    with st.sidebar:
        st.header("🎮 Player Dashboard")
        
        # Main stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("XP", st.session_state.xp)
        with col2:
            st.metric("Level", st.session_state.current_level)
        with col3:
            st.metric("⚛️", st.session_state.quantum_coins)
        
        # Progress to next level
        current_level_xp = st.session_state.xp % 100
        progress = current_level_xp / 100
        st.progress(progress)
        st.caption(f"{current_level_xp}/100 XP to next level")
        
        # Streak counter
        if st.session_state.quantum_streak > 0:
            st.markdown(f"🔥 **Quantum Streak:** {st.session_state.quantum_streak}")
        
        # Recent achievements
        if st.session_state.achievements:
            st.subheader("🏆 Recent Achievements")
            for achievement in st.session_state.achievements[-3:]:
                st.success(f"🎖️ {achievement}")
        
        # Mastery levels
        st.subheader("📈 Concept Mastery")
        for concept, level in st.session_state.mastery_levels.items():
            stars = "⭐" * level + "☆" * (3 - level)
            st.markdown(f"**{concept.title()}:** {stars}")
        
        st.divider()
        
        st.header("🗺️ Navigate")
        page = st.radio(
            "Choose a module:",
            [
                "🏰 Story Mode",
                "🎓 AI Quantum Tutor",
                "🧪 Superposition Lab",
                "⚙️ Quantum Gates Workshop", 
                "🌍 Bloch Explorer",
                "🧠 Quiz Center",
                "🎮 Quantum Challenges",
                "💎 Quantum Shop",
                "🚀 Future Quest Modules",
                "📚 Resources"
            ]
        )

    # --- Page Content ---
    if page == "🏰 Story Mode":
        story_mode_enhanced()
    elif page == "🎓 AI Quantum Tutor":
        ai_quantum_tutor()
    elif page == "🧪 Superposition Lab":
        superposition_lab_enhanced()
    elif page == "⚙️ Quantum Gates Workshop":
        quantum_gates_workshop_enhanced()
    elif page == "🌍 Bloch Explorer":
        bloch_explorer()
    elif page == "🧠 Quiz Center":
        run_simple_quiz()
    elif page == "🎮 Quantum Challenges":
        quantum_challenges()
    elif page == "💎 Quantum Shop":
        quantum_shop()
    elif page == "🚀 Future Quest Modules":
        future_quest_modules()
    elif page == "📚 Resources":
        resources_page()

# ----------------------- Main App Entry Point ----------------------------

if __name__ == '__main__':
    main_app()
