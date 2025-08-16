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
from qiskit import QuantumCircuit, execute
from qiskit.providers.aer import Aer
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# MathCraft: Quantum Quest — Enhanced Interactive Quantum Mechanics
# Now with comprehensive tutoring system and quantum reward mechanics
# -----------------------------------------------------------------------------

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
if 'interference_challenges' not in st.session_state:
    st.session_state.interference_challenges = {
        "max_constructive": False,
        "max_destructive": False,
        "half_prob": False
    }
if 'quantum_streak' not in st.session_state:
    st.session_state.quantum_streak = 0
if 'last_activity_date' not in st.session_state:
    st.session_state.last_activity_date = None
if 'mastery_levels' not in st.session_state:
    st.session_state.mastery_levels = {
        'superposition': 0,
        'gates': 0,
        'entanglement': 0,
        'measurement': 0,
        'interference': 0
    }

# ----------------------- Enhanced Quantum Concepts ----------------------------

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
    coin_reward = int(base_xp * multiplier * 0.5)  # Coins are half XP value
    
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
        st.session_state.quantum_coins += new_level * 10  # Level up bonus

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
                Think of a spinning coin in the air - before it lands, it's neither heads nor tails, but both! Measurement is like catching the coin.
                """,
                'examples': [
                    "Equal superposition: |+⟩ = (|0⟩ + |1⟩)/√2 - 50% chance each",
                    "Unequal superposition: 0.6|0⟩ + 0.8|1⟩ - 36% chance |0⟩, 64% chance |1⟩"
                ],
                'common_mistakes': [
                    "❌ Thinking the qubit 'chooses' a state before measurement",
                    "❌ Confusing probability amplitudes with probabilities",
                    "❌ Forgetting about the normalization condition"
                ]
            },
            'gates': {
                'title': '⚙️ Quantum Gates',
                'simple': "Quantum gates are operations that transform qubit states. They're like functions that take a quantum state as input and produce a new quantum state as output.",
                'detailed': """
                **Key Properties:**
                - Quantum gates are reversible (unitary operations)
                - They preserve the total probability (|α|² + |β|² = 1)
                - Gates can create, manipulate, or destroy superposition
                
                **Common Gates:**
                - **X Gate**: Quantum NOT - flips |0⟩ ↔ |1⟩
                - **H Gate**: Hadamard - creates superposition from basis states
                - **Z Gate**: Phase flip - adds minus sign to |1⟩ state
                - **CNOT**: Two-qubit gate for creating entanglement
                """,
                'examples': [
                    "H|0⟩ = (|0⟩ + |1⟩)/√2 - creates equal superposition",
                    "X|0⟩ = |1⟩ - classical bit flip",
                    "HH|0⟩ = |0⟩ - two Hadamards cancel out"
                ],
                'common_mistakes': [
                    "❌ Thinking gates 'measure' the qubit",
                    "❌ Forgetting gates are matrix operations",
                    "❌ Not considering the order of gate operations"
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
                - You cannot predict individual outcomes, only probabilities
                """,
                'examples': [
                    "Measuring |+⟩ gives |0⟩ or |1⟩ with 50% chance each",
                    "Measuring |0⟩ always gives |0⟩ (100% probability)",
                    "After measuring, the qubit is in the measured state"
                ],
                'common_mistakes': [
                    "❌ Thinking you can measure without disturbing the system",
                    "❌ Confusing probability amplitudes with probabilities",
                    "❌ Expecting deterministic outcomes from superposition states"
                ]
            },
            'entanglement': {
                'title': '🔗 Quantum Entanglement',
                'simple': "When qubits become entangled, they're connected in a spooky way - measuring one instantly affects the other, no matter how far apart they are!",
                'detailed': """
                **Bell States (Maximally Entangled):**
                - |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 - both qubits always agree
                - |Φ⁻⟩ = (|00⟩ - |11⟩)/√2 - both qubits always agree (with phase)
                - |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 - qubits always disagree
                - |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2 - qubits always disagree (with phase)
                
                **Properties:**
                - Cannot be written as a product of individual qubit states
                - Measuring one qubit instantly determines the other's state
                - The foundation of quantum computing's power
                """,
                'examples': [
                    "In |Φ⁺⟩: If first qubit measures |0⟩, second is definitely |0⟩",
                    "In |Ψ⁺⟩: If first qubit measures |0⟩, second is definitely |1⟩",
                    "Entanglement enables quantum teleportation and cryptography"
                ],
                'common_mistakes': [
                    "❌ Thinking entanglement allows faster-than-light communication",
                    "❌ Confusing correlation with causation",
                    "❌ Not understanding non-locality vs. non-communication"
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

# ----------------------- Enhanced Quiz System ----------------------------

class EnhancedQuizSystem:
    """Comprehensive quiz system with multiple question types and adaptive difficulty"""
    
    @staticmethod
    def get_quiz_questions():
        return {
            'superposition_basics': {
                'difficulty': 'easy',
                'concept': 'superposition',
                'questions': [
                    {
                        'question': "What does the Hadamard gate do to a |0⟩ qubit?",
                        'options': ["Flips it to |1⟩", "Creates an equal superposition", "Measures it", "Does nothing"],
                        'correct': 1,
                        'explanation': "The Hadamard gate creates an equal superposition: H|0⟩ = (|0⟩ + |1⟩)/√2"
                    },
                    {
                        'question': "In the state |ψ⟩ = 0.6|0⟩ + 0.8|1⟩, what's the probability of measuring |0⟩?",
                        'options': ["0.6", "0.36", "0.8", "0.5"],
                        'correct': 1,
                        'explanation': "Probability = |amplitude|². So P(|0⟩) = |0.6|² = 0.36"
                    },
                    {
                        'question': "Can a qubit be in a state (|0⟩ + |1⟩ + |2⟩)/√3?",
                        'options': ["Yes, this is valid", "No, qubits only have 2 states", "Yes, but only with measurement", "Only in quantum computers"],
                        'correct': 1,
                        'explanation': "Qubits have only two basis states: |0⟩ and |1⟩. A |2⟩ state doesn't exist for qubits."
                    }
                ]
            },
            'quantum_gates': {
                'difficulty': 'medium',
                'concept': 'gates',
                'questions': [
                    {
                        'question': "What happens when you apply two Hadamard gates to |0⟩ (i.e., HH|0⟩)?",
                        'options': ["You get |1⟩", "You get |0⟩", "You get (|0⟩ + |1⟩)/√2", "The state becomes undefined"],
                        'correct': 1,
                        'explanation': "H is its own inverse: HH = I. So HH|0⟩ = I|0⟩ = |0⟩"
                    },
                    {
                        'question': "Which gate adds a phase of -1 to the |1⟩ state while leaving |0⟩ unchanged?",
                        'options': ["X gate", "Y gate", "Z gate", "H gate"],
                        'correct': 2,
                        'explanation': "The Z gate does: Z|0⟩ = |0⟩ and Z|1⟩ = -|1⟩"
                    },
                    {
                        'question': "What's special about quantum gates compared to classical logic gates?",
                        'options': ["They're faster", "They're reversible", "They use less energy", "They work on superposition"],
                        'correct': 3,
                        'explanation': "Quantum gates work on superposition states and preserve quantum information, unlike classical gates."
                    }
                ]
            },
            'measurement_theory': {
                'difficulty': 'medium',
                'concept': 'measurement',
                'questions': [
                    {
                        'question': "After measuring a qubit in superposition, what happens to the state?",
                        'options': ["It stays in superposition", "It collapses to a definite state", "It becomes entangled", "It disappears"],
                        'correct': 1,
                        'explanation': "Measurement causes wavefunction collapse - the superposition is destroyed and the qubit is in a definite state."
                    },
                    {
                        'question': "You measure the state (|0⟩ + i|1⟩)/√2 in the computational basis. What are the probabilities?",
                        'options': ["P(0)=1, P(1)=0", "P(0)=0.5, P(1)=0.5", "P(0)=0, P(1)=1", "P(0)=i, P(1)=1"],
                        'correct': 1,
                        'explanation': "P(0) = |1/√2|² = 0.5, P(1) = |i/√2|² = 0.5. The phase doesn't affect measurement probabilities."
                    }
                ]
            },
            'entanglement_advanced': {
                'difficulty': 'hard',
                'concept': 'entanglement',
                'questions': [
                    {
                        'question': "Which Bell state represents (|00⟩ + |11⟩)/√2?",
                        'options': ["|Φ⁺⟩", "|Φ⁻⟩", "|Ψ⁺⟩", "|Ψ⁻⟩"],
                        'correct': 0,
                        'explanation': "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 is the classic Bell state where both qubits are always correlated."
                    },
                    {
                        'question': "If you measure the first qubit of |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 and get |0⟩, what's the state of the second qubit?",
                        'options': ["|0⟩", "|1⟩", "(|0⟩ + |1⟩)/√2", "Cannot be determined"],
                        'correct': 1,
                        'explanation': "In |Ψ⁺⟩, the qubits are anticorrelated. If first is |0⟩, second must be |1⟩."
                    }
                ]
            }
        }

def run_enhanced_quiz():
    """Enhanced quiz interface with adaptive difficulty and detailed feedback"""
    
    st.header("🧠 Enhanced Quantum Quiz System")
    
    quiz_data = EnhancedQuizSystem.get_quiz_questions()
    
    # Quiz selection
    quiz_name = st.selectbox(
        "Choose a quiz topic:",
        list(quiz_data.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    selected_quiz = quiz_data[quiz_name]
    
    # Show concept tutorial first
    if st.expander(f"📚 Learn about {selected_quiz['concept'].title()} first"):
        show_concept_tutor(selected_quiz['concept'])
    
    st.markdown(f"**Difficulty:** {selected_quiz['difficulty'].title()}")
    st.markdown(f"**Questions:** {len(selected_quiz['questions'])}")
    
    if f"quiz_{quiz_name}" not in st.session_state:
        st.session_state[f"quiz_{quiz_name}"] = {
            'current_question': 0,
            'score': 0,
            'answers': [],
            'completed': False
        }
    
    quiz_state = st.session_state[f"quiz_{quiz_name}"]
    questions = selected_quiz['questions']
    
    if not quiz_state['completed']:
        current_q = quiz_state['current_question']
        
        if current_q < len(questions):
            question_data = questions[current_q]
            
            st.subheader(f"Question {current_q + 1} of {len(questions)}")
            st.markdown(f"**{question_data['question']}**")
            
            # Display options
            answer = st.radio(
                "Choose your answer:",
                question_data['options'],
                key=f"q_{quiz_name}_{current_q}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Submit Answer", type="primary"):
                    selected_challenge = st.selectbox("Choose a challenge:", challenges, format_func=lambda x: x['name'])
        
        st.markdown(f"**Challenge:** {selected_challenge['description']}")
        st.markdown(f"**Starting state:** {selected_challenge['initial_state']}")
        st.markdown(f"**Target:** {selected_challenge['expected']}")
        
        # Let user build the solution
        if selected_challenge['gates'] == ["?"]:
            user_gate = st.selectbox("Choose your gate:", ["X", "H", "Z", "Y"])
            
            if st.button("Test Solution"):
                if selected_challenge['name'] == "Create |1⟩" and user_gate == "X":
                    st.success("🎉 Correct! X gate flips |0⟩ to |1⟩")
                    add_xp_enhanced(20, "Solved gate challenge!", 'medium', True)
                elif selected_challenge['name'] == "Superposition Creation" and user_gate == "H":
                    st.success("🎉 Correct! Hadamard creates superposition!")
                    add_xp_enhanced(20, "Solved gate challenge!", 'medium', True)
                else:
                    st.error("❌ Not quite right. Think about what each gate does!")
                    add_xp_enhanced(5, "Keep trying!", 'medium', False)
    
    # Circuit builder mode
    else:  # Circuit Builder
        st.subheader("Quantum Circuit Builder")
        
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
            
            st.subheader("2. Build Your Circuit")
            
            # Gate sequence builder
            if 'gate_sequence' not in st.session_state:
                st.session_state.gate_sequence = []
            
            col_add, col_clear = st.columns(2)
            
            with col_add:
                new_gate = st.selectbox("Add gate:", ["X", "Y", "Z", "H", "Rx", "Ry", "Rz", "Phase"])
                
                if new_gate in ["Rx", "Ry", "Rz", "Phase"]:
                    angle = st.slider(f"{new_gate} angle (radians)", 0.0, 2*np.pi, np.pi/2, 0.1)
                    gate_label = f"{new_gate}({angle:.2f})"
                else:
                    angle = None
                    gate_label = new_gate
                
                if st.button("Add Gate"):
                    st.session_state.gate_sequence.append((new_gate, angle, gate_label))
            
            with col_clear:
                if st.button("Clear Circuit"):
                    st.session_state.gate_sequence = []
            
            # Display current circuit
            if st.session_state.gate_sequence:
                circuit_display = " → ".join([gate[2] for gate in st.session_state.gate_sequence])
                st.markdown(f"**Current circuit:** {init_state} → {circuit_display}")
            else:
                st.markdown(f"**Current circuit:** {init_state}")
            
            # Apply gates and show result
            final_state = state.copy()
            for gate_info in st.session_state.gate_sequence:
                gate_name, angle, _ = gate_info
                
                if gate_name == "X":
                    final_state = apply(X, final_state)
                elif gate_name == "Y":
                    final_state = apply(Y, final_state)
                elif gate_name == "Z":
                    final_state = apply(Z, final_state)
                elif gate_name == "H":
                    final_state = apply(H, final_state)
                elif gate_name == "Rx":
                    final_state = apply(Rx(angle), final_state)
                elif gate_name == "Ry":
                    final_state = apply(Ry(angle), final_state)
                elif gate_name == "Rz":
                    final_state = apply(Rz(angle), final_state)
                elif gate_name == "Phase":
                    final_state = apply(phase(angle), final_state)
            
            st.subheader("3. Results")
            
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
            
            # Circuit simulation button
            if st.button("🎯 Simulate Circuit", type="primary"):
                add_xp_enhanced(15, "Simulated quantum circuit!", 'medium', True)
                st.success("Circuit simulation completed!")
        
        with col2:
            x, y, z = bloch_coords(final_state)
            fig = create_bloch_sphere(x, y, z, "Final State on Bloch Sphere")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🧠 Gate Reference")
            with st.expander("Quick Gate Guide"):
                st.markdown("""
                - **X**: Bit flip (|0⟩ ↔ |1⟩)
                - **H**: Creates superposition
                - **Z**: Phase flip (adds minus to |1⟩)
                - **Rx/Ry/Rz**: Rotation around X/Y/Z axis
                - **Phase**: Adds phase to |1⟩ state
                """)

# ----------------------- Enhanced Quiz Center ----------------------------

def enhanced_quiz_center():
    st.header("🧠 Enhanced Quantum Quiz Center")
    st.markdown("Test your quantum knowledge with adaptive quizzes and instant feedback!")
    
    # Quiz selection with difficulty indicators
    col1, col2 = st.columns([2, 1])
    
    with col1:
        quiz_categories = {
            'Fundamentals': {
                'superposition_basics': ('Superposition Basics', 'easy', '🌱'),
                'measurement_intro': ('Measurement Introduction', 'easy', '🌱'),
            },
            'Intermediate': {
                'quantum_gates': ('Quantum Gates', 'medium', '🔬'),
                'bloch_sphere': ('Bloch Sphere', 'medium', '🔬'),
            },
            'Advanced': {
                'entanglement_advanced': ('Advanced Entanglement', 'hard', '🎓'),
                'quantum_algorithms': ('Quantum Algorithms', 'hard', '🎓'),
            }
        }
        
        st.subheader("📚 Choose Your Quiz")
        
        for category, quizzes in quiz_categories.items():
            with st.expander(f"{category} Quizzes"):
                for quiz_id, (name, difficulty, icon) in quizzes.items():
                    col_quiz, col_diff, col_button = st.columns([3, 1, 1])
                    
                    with col_quiz:
                        st.markdown(f"**{icon} {name}**")
                    
                    with col_diff:
                        difficulty_colors = {'easy': '🟢', 'medium': '🟡', 'hard': '🔴'}
                        st.markdown(f"{difficulty_colors[difficulty]} {difficulty.title()}")
                    
                    with col_button:
                        if st.button(f"Start", key=f"start_{quiz_id}"):
                            st.session_state.selected_quiz = quiz_id
        
        # Run selected quiz
        if 'selected_quiz' in st.session_state:
            run_enhanced_quiz()
    
    with col2:
        st.subheader("📊 Your Quiz Statistics")
        
        # Overall stats
        total_quizzes = len([key for key in st.session_state.quiz_scores.keys()])
        if total_quizzes > 0:
            avg_score = np.mean(list(st.session_state.quiz_scores.values()))
            st.metric("Quizzes Completed", total_quizzes)
            st.metric("Average Score", f"{avg_score:.1f}%")
        else:
            st.info("Complete your first quiz to see statistics!")
        
        # Recent performance
        if st.session_state.quiz_scores:
            st.subheader("📈 Recent Scores")
            for quiz_name, score in list(st.session_state.quiz_scores.items())[-5:]:
                st.markdown(f"• {quiz_name}: {score:.0f}%")

# ----------------------- Quantum Challenges ----------------------------

def quantum_challenges():
    st.header("🎮 Quantum Challenges")
    st.markdown("Test your skills with these quantum mechanics challenges!")
    
    # Challenge categories
    challenge_type = st.selectbox(
        "Choose challenge type:",
        ["🎯 State Creation", "🔧 Circuit Puzzles", "📊 Measurement Games", "🧩 Quantum Logic"]
    )
    
    if challenge_type == "🎯 State Creation":
        st.subheader("State Creation Challenges")
        st.markdown("Create specific quantum states using the minimum number of operations!")
        
        challenges = [
            {
                "name": "The Perfect Balance",
                "description": "Create a state with exactly 30% probability of measuring |0⟩",
                "target_prob": 0.3,
                "max_operations": 3,
                "difficulty": "medium",
                "reward": 25
            },
            {
                "name": "Complex Phase Master",
                "description": "Create the state (|0⟩ + e^(iπ/3)|1⟩)/√2",
                "target_state": "complex_phase",
                "max_operations": 2,
                "difficulty": "hard",
                "reward": 40
            },
            {
                "name": "Quantum Coin Flip",
                "description": "Create perfect 50/50 superposition starting from |1⟩",
                "initial_state": "|1⟩",
                "target_prob": 0.5,
                "max_operations": 1,
                "difficulty": "easy",
                "reward": 15
            }
        ]
        
        selected_challenge = st.selectbox("Choose a challenge:", challenges, format_func=lambda x: x['name'])
        
        st.markdown(f"**🎯 Challenge:** {selected_challenge['name']}")
        st.markdown(f"**📝 Description:** {selected_challenge['description']}")
        st.markdown(f"**⚡ Difficulty:** {selected_challenge['difficulty'].title()}")
        st.markdown(f"**🏆 Reward:** {selected_challenge['reward']} XP + coins")
        st.markdown(f"**🔢 Max Operations:** {selected_challenge['max_operations']}")
        
        # Challenge interface
        if 'challenge_operations' not in st.session_state:
            st.session_state.challenge_operations = []
        
        st.subheader("🔧 Your Solution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            operation = st.selectbox("Add operation:", ["H", "X", "Y", "Z", "Rx(π/2)", "Ry(π/2)", "Rz(π/2)"])
            
            if st.button("Add Operation"):
                if len(st.session_state.challenge_operations) < selected_challenge['max_operations']:
                    st.session_state.challenge_operations.append(operation)
                else:
                    st.error(f"Maximum {selected_challenge['max_operations']} operations allowed!")
        
        with col2:
            if st.button("Clear Solution"):
                st.session_state.challenge_operations = []
        
        # Show current solution
        if st.session_state.challenge_operations:
            operations_str = " → ".join(st.session_state.challenge_operations)
            initial = selected_challenge.get('initial_state', '|0⟩')
            st.markdown(f"**Current solution:** {initial} → {operations_str}")
        
        # Test solution
        if st.button("🧪 Test Solution", type="primary"):
            if not st.session_state.challenge_operations:
                st.error("Add at least one operation!")
                return
            
            # Simulate the operations
            initial_state = ket1() if selected_challenge.get('initial_state') == '|1⟩' else ket0()
            current_state = initial_state.copy()
            
            for op in st.session_state.challenge_operations:
                if op == "H":
                    current_state = apply(H, current_state)
                elif op == "X":
                    current_state = apply(X, current_state)
                elif op == "Y":
                    current_state = apply(Y, current_state)
                elif op == "Z":
                    current_state = apply(Z, current_state)
                elif op == "Rx(π/2)":
                    current_state = apply(Rx(np.pi/2), current_state)
                elif op == "Ry(π/2)":
                    current_state = apply(Ry(np.pi/2), current_state)
                elif op == "Rz(π/2)":
                    current_state = apply(Rz(np.pi/2), current_state)
            
            probs = measure_probs(current_state)
            
            # Check if challenge is solved
            success = False
            if 'target_prob' in selected_challenge:
                error = abs(probs['0'] - selected_challenge['target_prob'])
                if error < 0.01:
                    success = True
                    st.success(f"🎉 Challenge completed! Error: {error:.4f}")
                else:
                    st.error(f"❌ Not quite right. Error: {error:.3f}")
            
            # Show final state
            st.code(f"Final state: {cfmt(current_state[0,0])}|0⟩ + {cfmt(current_state[1,0])}|1⟩")
            st.markdown(f"P(|0⟩) = {probs['0']:.4f}, P(|1⟩) = {probs['1']:.4f}")
            
            if success:
                reward = selected_challenge['reward']
                difficulty = selected_challenge['difficulty']
                add_xp_enhanced(reward, f"Completed {selected_challenge['name']}!", difficulty, True)
                
                # Special achievement for perfect solutions
                if len(st.session_state.challenge_operations) == 1 and selected_challenge['max_operations'] > 1:
                    st.balloons()
                    st.success("🌟 PERFECT SOLUTION BONUS! +50% XP")
                    add_xp_enhanced(reward // 2, "Perfect solution bonus!", difficulty, True)

# ----------------------- Quantum Shop ----------------------------

def quantum_shop():
    st.header("💎 Quantum Shop")
    st.markdown("Spend your quantum coins on useful upgrades and cosmetics!")
    
    st.markdown(f"**Your Quantum Coins:** ⚛️ {st.session_state.quantum_coins}")
    
    if 'shop_inventory' not in st.session_state:
        st.session_state.shop_inventory = {
            'hint_booster': 0,
            'xp_multiplier': 0,
            'theme_dark': False,
            'theme_neon': False,
            'advanced_bloch': False
        }
    
    # Shop categories
    shop_category = st.selectbox("Shop Category:", ["🔧 Utilities", "🎨 Themes", "📊 Visualizations"])
    
    if shop_category == "🔧 Utilities":
        st.subheader("Utility Items")
        
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
    
    elif shop_category == "🎨 Themes":
        st.subheader("Visual Themes")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌙 Dark Quantum Theme")
            st.markdown("Sleek dark theme for night studying")
            st.markdown("**Price:** ⚛️ 150")
            
            if st.session_state.shop_inventory['theme_dark']:
                st.success("✅ Owned")
            else:
                if st.button("Buy Dark Theme"):
                    if st.session_state.quantum_coins >= 150:
                        st.session_state.quantum_coins -= 150
                        st.session_state.shop_inventory['theme_dark'] = True
                        st.success("Theme purchased! Restart to apply.")
                    else:
                        st.error("Not enough quantum coins!")
        
        with col2:
            st.markdown("### 🌈 Neon Quantum Theme")
            st.markdown("Vibrant neon colors")
            st.markdown("**Price:** ⚛️ 200")
            
            if st.session_state.shop_inventory['theme_neon']:
                st.success("✅ Owned")
            else:
                if st.button("Buy Neon Theme"):
                    if st.session_state.quantum_coins >= 200:
                        st.session_state.quantum_coins -= 200
                        st.session_state.shop_inventory['theme_neon'] = True
                        st.success("Theme purchased! Restart to apply.")
                    else:
                        st.error("Not enough quantum coins!")
    
    else:  # Visualizations
        st.subheader("Advanced Visualizations")
        
        st.markdown("### 🌐 Advanced Bloch Sphere")
        st.markdown("Enhanced Bloch sphere with animations and multiple qubits")
        st.markdown("**Price:** ⚛️ 300")
        
        if st.session_state.shop_inventory['advanced_bloch']:
            st.success("✅ Owned - Available in all labs!")
        else:
            if st.button("Buy Advanced Bloch Sphere"):
                if st.session_state.quantum_coins >= 300:
                    st.session_state.quantum_coins -= 300
                    st.session_state.shop_inventory['advanced_bloch'] = True
                    st.success("Purchased! Advanced visualizations now available!")
                else:
                    st.error("Not enough quantum coins!")

# ----------------------- Remaining Original Functions ----------------------------
# (Keep the original functions that haven't been enhanced)

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
                add_xp_enhanced(20, "Achieved perfect constructive interference!", 'medium', True)
            else:
                st.error("Hint: Make the two waves perfectly in phase.")

    with col_c2:
        st.info("Challenge 2: Perfect Destructive Interference")
        if st.button("Check 2"):
            if abs(final_prob) < 0.01:
                st.session_state.interference_challenges["max_destructive"] = True
                add_xp_enhanced(20, "Achieved perfect destructive interference!", 'medium', True)
            else:
                st.error("Hint: The waves need to cancel each other out.")

    with col_c3:
        st.info("Challenge 3: 50/50 Chance")
        if st.button("Check 3"):
            if abs(final_prob - 1) < 0.01:
                st.session_state.interference_challenges["half_prob"] = True
                add_xp_enhanced(20, "Created a 50/50 probability!", 'medium', True)
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
    
        add_xp_enhanced(10, "Simulated Stern-Gerlach experiment.", 'easy', True)

def entanglement_workshop():
    st.header("Quantum Entanglement Lab")
    st.markdown("What Einstein called **'spooky action at a distance.'** When two qubits are entangled, their fates are linked.")
    st.markdown("Measuring one instantly affects the other, no matter the distance between them.")
    
    st.subheader("Bell State Generation")
    st.write("We will create a Bell state, $|Φ^+\\rangle = (|00\\rangle + |11\\rangle)/\\sqrt{2}$.")
    st.write("This state means the two qubits are perfectly correlated. If the first qubit is measured as $|0\\rangle$, the second will also be $|0\\rangle$. The same is true for $|1\\rangle$.")
    
    qc2 = QuantumCircuit(2)
    st.markdown("#### Quantum Circuit")
    st.code("H-gate on qubit 0, then CNOT(0,1)")
    
    shots = st.slider("Number of measurements", 100, 2048, 1024)
    
    if st.button("🔬 Entangle & Measure", type="primary"):
        with st.spinner("Executing entanglement experiment..."):
            qc2.h(0)
            qc2.cx(0, 1)
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
                add_xp_enhanced(30, "Confirmed spooky action! No uncorrelated outcomes.", 'hard', True)
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
    st.code("Circuit diagram would be displayed here")
    
    if st.button("Run Simulation", type="primary"):
        if not qc.data:
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
    
            add_xp_enhanced(15, "Successfully ran a quantum simulation!", 'medium', True)

def future_quest_modules():
    st.header("🚀 Future Quest Modules")
    st.info("Stay tuned! More advanced modules are coming soon.")
    
    # Preview upcoming features
    upcoming_features = [
        {
            "title": "🌐 Quantum Teleportation Lab",
            "description": "Learn how to transfer a quantum state from one qubit to another using entanglement.",
            "difficulty": "Advanced",
            "eta": "Coming Soon"
        },
        {
            "title": "🔐 Quantum Cryptography Workshop", 
            "description": "Explore quantum key distribution and unbreakable quantum communication.",
            "difficulty": "Expert",
            "eta": "Q2 2024"
        },
        {
            "title": "🔍 Grover's Search Algorithm",
            "description": "Use quantum algorithms to search databases faster than any classical computer.",
            "difficulty": "Expert", 
            "eta": "Q3 2024"
        },
        {
            "title": "🧮 Shor's Factoring Algorithm",
            "description": "Discover how quantum computers could break current encryption methods.",
            "difficulty": "Master",
            "eta": "Q4 2024"
        },
        {
            "title": "🎮 Quantum Games & Puzzles",
            "description": "Fun quantum mechanical games and brain teasers.",
            "difficulty": "All Levels",
            "eta": "Next Update"
        }
    ]
    
    for feature in upcoming_features:
        with st.expander(f"{feature['title']} - {feature['eta']}"):
            st.markdown(f"**Description:** {feature['description']}")
            st.markdown(f"**Difficulty:** {feature['difficulty']}")
            st.markdown(f"**Expected Release:** {feature['eta']}")
    
    # User feedback section
    st.subheader("📝 Request Features")
    st.markdown("What would you like to see in future updates?")
    
    user_request = st.text_area("Describe a feature you'd like to see:")
    if st.button("Submit Request"):
        if user_request:
            st.success("Thank you for your feedback! We'll consider this for future updates.")
            add_xp_enhanced(5, "Provided valuable feedback!", 'easy', True)

def resources_page():
    st.header("📚 Quantum Learning Resources")
    st.markdown("### Continue Your Quantum Journey")
    
    # Resource categories
    resource_type = st.selectbox(
        "Choose resource type:",
        ["📖 Books & Textbooks", "🎥 Videos & Courses", "💻 Interactive Tools", "🔬 Research Papers"]
    )
    
    if resource_type == "📖 Books & Textbooks":
        st.subheader("Recommended Reading")
        
        books = [
            {
                "title": "Quantum Computing: An Applied Approach",
                "authors": "Hidary, J. D.",
                "level": "Beginner to Intermediate",
                "description": "Practical introduction with hands-on examples"
            },
            {
                "title": "Quantum Computation and Quantum Information", 
                "authors": "Nielsen, M. A. & Chuang, I. L.",
                "level": "Intermediate to Advanced",
                "description": "The definitive textbook on quantum information"
            },
            {
                "title": "Programming Quantum Computers",
                "authors": "Johnston, E. R., Harrigan, N., & Gimeno-Segovia, M.",
                "level": "Intermediate",
                "description": "Learn to program quantum algorithms"
            }
        ]
        
        for book in books:
            st.markdown(f"**{book['title']}**")
            st.markdown(f"*Authors:* {book['authors']}")
            st.markdown(f"*Level:* {book['level']}")
            st.markdown(f"*Description:* {book['description']}")
            st.markdown("---")
    
    elif resource_type == "🎥 Videos & Courses":
        st.subheader("Video Learning")
        
        courses = [
            {
                "title": "IBM Qiskit Textbook",
                "provider": "IBM Quantum",
                "url": "https://qiskit.org/textbook/",
                "description": "Comprehensive online textbook with interactive examples"
            },
            {
                "title": "Quantum Mechanics and Quantum Computation",
                "provider": "UC Berkeley (edX)",
                "description": "University-level course on quantum mechanics foundations"
            },
            {
                "title": "Microsoft Quantum Development Kit",
                "provider": "Microsoft",
                "description": "Learn Q# programming language for quantum computing"
            }
        ]
        
        for course in courses:
            st.markdown(f"**{course['title']}**")
            st.markdown(f"*Provider:* {course['provider']}")
            st.markdown(f"*Description:* {course['description']}")
            if 'url' in course:
                st.markdown(f"*Link:* {course['url']}")
            st.markdown("---")
    
    elif resource_type == "💻 Interactive Tools":
        st.subheader("Hands-on Learning Platforms")
        
        tools = [
            {
                "name": "IBM Quantum Experience",
                "description": "Run quantum circuits on real quantum computers",
                "access": "Free with registration"
            },
            {
                "name": "Google Cirq",
                "description": "Python library for quantum circuits on Google's quantum processors",
                "access": "Open source"
            },
            {
                "name": "Microsoft Q# Simulator",
                "description": "Simulate quantum algorithms locally",
                "access": "Free download"
            },
            {
                "name": "Quantum Computing Playground",
                "description": "Browser-based quantum circuit simulator",
                "access": "Free online"
            }
        ]
        
        for tool in tools:
            st.markdown(f"**{tool['name']}**")
            st.markdown(f"*Description:* {tool['description']}")
            st.markdown(f"*Access:* {tool['access']}")
            st.markdown("---")
    
    else:  # Research Papers
        st.subheader("Key Research Papers")
        st.info("These papers shaped the field of quantum computing")
        
        papers = [
            {
                "title": "Quantum Theory, the Church-Turing Principle and the Universal Quantum Computer",
                "author": "David Deutsch (1985)",
                "significance": "Laid theoretical foundations for quantum computing"
            },
            {
                "title": "Algorithms for Quantum Computation: Discrete Logarithms and Factoring",
                "author": "Peter Shor (1994)",
                "significance": "Showed quantum computers could break current encryption"
            },
            {
                "title": "Quantum Mechanics Helps in Searching for a Needle in a Haystack",
                "author": "Lov Grover (1996)",
                "significance": "Demonstrated quantum speedup for search problems"
            }
        ]
        
        for paper in papers:
            st.markdown(f"**{paper['title']}**")
            st.markdown(f"*Author:* {paper['author']}")
            st.markdown(f"*Significance:* {paper['significance']}")
            st.markdown("---")
    
    # Study tips
    st.subheader("🎯 Study Tips")
    study_tips = [
        "Start with the mathematical foundations - linear algebra and complex numbers",
        "Practice with simulators before trying real quantum hardware",
        "Join quantum computing communities and forums for discussions",
        "Work through problems step-by-step, don't rush concepts",
        "Relate quantum concepts to classical analogs when possible",
        "Experiment with different quantum programming languages"
    ]
    
    for i, tip in enumerate(study_tips, 1):
        st.markdown(f"{i}. {tip}")

# ----------------------- Quiz System Implementation ----------------------------

def enhanced_quiz_center():
    st.header("🧠 Enhanced Quantum Quiz Center")
    st.markdown("Test your quantum knowledge with adaptive quizzes and instant feedback!")
    
    # Define all quiz questions
    all_quizzes = {
        'superposition_basics': {
            'title': 'Superposition Fundamentals',
            'difficulty': 'easy',
            'concept': 'superposition',
            'questions': [
                {
                    'question': "What does the Hadamard gate do to a |0⟩ qubit?",
                    'options': ["Flips it to |1⟩", "Creates an equal superposition", "Measures it", "Does nothing"],
                    'correct': 1,
                    'explanation': "The Hadamard gate creates an equal superposition: H|0⟩ = (|0⟩ + |1⟩)/√2"
                },
                {
                    'question': "In the state |ψ⟩ = 0.6|0⟩ + 0.8|1⟩, what's the probability of measuring |0⟩?",
                    'options': ["0.6", "0.36", "0.8", "0.5"],
                    'correct': 1,
                    'explanation': "Probability = |amplitude|². So P(|0⟩) = |0.6|² = 0.36"
                },
                {
                    'question': "Can a qubit be in a state (|0⟩ + |1⟩ + |2⟩)/√3?",
                    'options': ["Yes, this is valid", "No, qubits only have 2 states", "Yes, but only with measurement", "Only in quantum computers"],
                    'correct': 1,
                    'explanation': "Qubits have only two basis states: |0⟩ and |1⟩. A |2⟩ state doesn't exist for qubits."
                },
                {
                    'question': "What happens to a qubit in superposition when measured?",
                    'options': ["It stays in superposition", "It collapses to a definite state", "It becomes entangled", "It disappears"],
                    'correct': 1,
                    'explanation': "Measurement causes the superposition to collapse into one of the basis states."
                },
                {
                    'question': "Which state represents maximum superposition on the Bloch sphere?",
                    'options': ["North pole", "South pole", "Equator", "Center"],
                    'correct': 2,
                    'explanation': "States on the equator represent maximum superposition between |0⟩ and |1⟩."
                }
            ]
        },
        'quantum_gates': {
            'title': 'Quantum Gates Mastery',
            'difficulty': 'medium',
            'concept': 'gates',
            'questions': [
                {
                    'question': "What happens when you apply two Hadamard gates to |0⟩ (i.e., HH|0⟩)?",
                    'options': ["You get |1⟩", "You get |0⟩", "You get (|0⟩ + |1⟩)/√2", "The state becomes undefined"],
                    'correct': 1,
                    'explanation': "H is its own inverse: HH = I. So HH|0⟩ = I|0⟩ = |0⟩"
                },
                {
                    'question': "Which gate adds a phase of -1 to the |1⟩ state while leaving |0⟩ unchanged?",
                    'options': ["X gate", "Y gate", "Z gate", "H gate"],
                    'correct': 2,
                    'explanation': "The Z gate does: Z|0⟩ = |0⟩ and Z|1⟩ = -|1⟩"
                },
                {
                    'question': "What's the effect of applying X then Z to |0⟩?",
                    'options': ["|0⟩", "-|0⟩", "|1⟩", "-|1⟩"],
                    'correct': 3,
                    'explanation': "X|0⟩ = |1⟩, then Z|1⟩ = -|1⟩"
                },
                {
                    'question': "Which property must all quantum gates satisfy?",
                    'options': ["They must be Hermitian", "They must be unitary", "They must be diagonal", "They must be real"],
                    'correct': 1,
                    'explanation': "Quantum gates must be unitary to preserve probability and ensure reversibility."
                },
                {
                    'question': "What does the CNOT gate do?",
                    'options': ["Flips both qubits", "Flips target if control is |1⟩", "Creates superposition", "Measures the qubits"],
                    'correct': 1,
                    'explanation': "CNOT flips the target qubit if and only if the control qubit is in state |1⟩."
                }
            ]
        },
        'measurement_theory': {
            'title': 'Quantum Measurement',
            'difficulty': 'medium',
            'concept': 'measurement',
            'questions': [
                {
                    'question': "After measuring a qubit in superposition, what happens to the state?",
                    'options': ["It stays in superposition", "It collapses to a definite state", "It becomes entangled", "It disappears"],
                    'correct': 1,
                    'explanation': "Measurement causes wavefunction collapse - the superposition is destroyed and the qubit is in a definite state."
                },
                {
                    'question': "You measure the state (|0⟩ + i|1⟩)/√2 in the computational basis. What are the probabilities?",
                    'options': ["P(0)=1, P(1)=0", "P(0)=0.5, P(1)=0.5", "P(0)=0, P(1)=1", "Cannot be determined"],
                    'correct': 1,
                    'explanation': "P(0) = |1/√2|² = 0.5, P(1) = |i/√2|² = 0.5. The phase doesn't affect measurement probabilities."
                },
                {
                    'question': "What information is lost during quantum measurement?",
                    'options': ["Energy information", "Phase relationships", "Particle mass", "Nothing is lost"],
                    'correct': 1,
                    'explanation': "Measurement destroys phase relationships and superposition, keeping only probability information."
                },
                {
                    'question': "Can you determine the exact quantum state from a single measurement?",
                    'options': ["Yes, always", "No, never", "Only for basis states", "Only with special equipment"],
                    'correct': 2,
                    'explanation': "A single measurement only gives you one outcome. You need many measurements to estimate probabilities."
                }
            ]
        },
        'entanglement_advanced': {
            'title': 'Quantum Entanglement',
            'difficulty': 'hard',
            'concept': 'entanglement',
            'questions': [
                {
                    'question': "Which Bell state represents (|00⟩ + |11⟩)/√2?",
                    'options': ["|Φ⁺⟩", "|Φ⁻⟩", "|Ψ⁺⟩", "|Ψ⁻⟩"],
                    'correct': 0,
                    'explanation': "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2 is the classic Bell state where both qubits are always correlated."
                },
                {
                    'question': "If you measure the first qubit of |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2 and get |0⟩, what's the state of the second qubit?",
                    'options': ["|0⟩", "|1⟩", "(|0⟩ + |1⟩)/√2", "Cannot be determined"],
                    'correct': 1,
                    'explanation': "In |Ψ⁺⟩, the qubits are anticorrelated. If first is |0⟩, second must be |1⟩."
                },
                {
                    'question': "What makes the Bell states special?",
                    'options': ["They are separable", "They are maximally entangled", "They are easy to create", "They last forever"],
                    'correct': 1,
                    'explanation': "Bell states are maximally entangled - they cannot be written as a product of individual qubit states."
                },
                {
                    'question': "How do you create the |Φ⁺⟩ Bell state starting from |00⟩?",
                    'options': ["H on first, then CNOT", "CNOT then H on first", "H on both qubits", "X on second, then H on first"],
                    'correct': 0,
                    'explanation': "Apply Hadamard to first qubit: (|0⟩+|1⟩)|0⟩/√2, then CNOT: (|00⟩+|11⟩)/√2"
                }
            ]
        },
        'bloch_sphere': {
            'title': 'Bloch Sphere Geometry',
            'difficulty': 'medium',
            'concept': 'superposition',
            'questions': [
                {
                    'question': "Where is the |1⟩ state located on the Bloch sphere?",
                    'options': ["North pole", "South pole", "Equator", "Center"],
                    'correct': 1,
                    'explanation': "The |1⟩ state is at the south pole of the Bloch sphere (z = -1)."
                },
                {
                    'question': "What do states on the equator of the Bloch sphere represent?",
                    'options': ["Pure states", "Mixed states", "Equal superposition", "Entangled states"],
                    'correct': 2,
                    'explanation': "Equatorial states have equal probabilities for |0⟩ and |1⟩ measurements."
                },
                {
                    'question': "How many real parameters are needed to specify a qubit state on the Bloch sphere?",
                    'options': ["1", "2", "3", "4"],
                    'correct': 1,
                    'explanation': "Two angles (θ and φ) specify any point on the Bloch sphere."
                }
            ]
        },
        'quantum_algorithms': {
            'title': 'Quantum Algorithms',
            'difficulty': 'hard',
            'concept': 'gates',
            'questions': [
                {
                    'question': "What advantage does Grover's algorithm provide?",
                    'options': ["Exponential speedup", "Quadratic speedup", "Linear speedup", "No speedup"],
                    'correct': 1,
                    'explanation': "Grover's algorithm provides quadratic speedup for unstructured search problems."
                },
                {
                    'question': "What problem does Shor's algorithm solve efficiently?",
                    'options': ["Graph coloring", "Integer factorization", "Protein folding", "Weather prediction"],
                    'correct': 1,
                    'explanation': "Shor's algorithm efficiently factors large integers, threatening current cryptography."
                },
                {
                    'question': "What is quantum parallelism?",
                    'options': ["Running multiple quantum computers", "Using superposition to evaluate functions on multiple inputs", "Parallel gate operations", "Measuring multiple qubits"],
                    'correct': 1,
                    'explanation': "Quantum parallelism uses superposition to evaluate a function on many inputs simultaneously."
                }
            ]
        }
    }
    
    # Quiz interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Quiz selection
        st.subheader("📚 Available Quizzes")
        
        # Group quizzes by difficulty
        easy_quizzes = {k: v for k, v in all_quizzes.items() if v['difficulty'] == 'easy'}
        medium_quizzes = {k: v for k, v in all_quizzes.items() if v['difficulty'] == 'medium'}
        hard_quizzes = {k: v for k, v in all_quizzes.items() if v['difficulty'] == 'hard'}
        
        # Beginner section
        if easy_quizzes:
            st.markdown("### 🌱 Beginner Level")
            for quiz_id, quiz_data in easy_quizzes.items():
                col_name, col_diff, col_start = st.columns([3, 1, 1])
                with col_name:
                    st.markdown(f"**{quiz_data['title']}**")
                with col_diff:
                    st.markdown("🟢 Easy")
                with col_start:
                    if st.button("Start", key=f"start_{quiz_id}"):
                        st.session_state.current_quiz = quiz_id
                        st.experimental_rerun()
        
        # Intermediate section
        if medium_quizzes:
            st.markdown("### 🔬 Intermediate Level")
            for quiz_id, quiz_data in medium_quizzes.items():
                col_name, col_diff, col_start = st.columns([3, 1, 1])
                with col_name:
                    st.markdown(f"**{quiz_data['title']}**")
                with col_diff:
                    st.markdown("🟡 Medium")
                with col_start:
                    if st.button("Start", key=f"start_{quiz_id}"):
                        st.session_state.current_quiz = quiz_id
                        st.experimental_rerun()
        
        # Advanced section
        if hard_quizzes:
            st.markdown("### 🎓 Advanced Level")
            for quiz_id, quiz_data in hard_quizzes.items():
                col_name, col_diff, col_start = st.columns([3, 1, 1])
                with col_name:
                    st.markdown(f"**{quiz_data['title']}**")
                with col_diff:
                    st.markdown("🔴 Hard")
                with col_start:
                    if st.button("Start", key=f"start_{quiz_id}"):
                        st.session_state.current_quiz = quiz_id
                        st.experimental_rerun()
        
        # Run active quiz
        if 'current_quiz' in st.session_state and st.session_state.current_quiz in all_quizzes:
            st.markdown("---")
            run_active_quiz(all_quizzes[st.session_state.current_quiz], st.session_state.current_quiz)
    
    with col2:
        st.subheader("📊 Quiz Statistics")
        
        # Overall performance
        if st.session_state.quiz_scores:
            scores = list(st.session_state.quiz_scores.values())
            avg_score = np.mean(scores)
            best_score = max(scores)
            
            st.metric("Quizzes Completed", len(scores))
            st.metric("Average Score", f"{avg_score:.1f}%")
            st.metric("Best Score", f"{best_score:.1f}%")
            
            # Performance by concept
            st.subheader("📈 Mastery Levels")
            for concept, level in st.session_state.mastery_levels.items():
                stars = "⭐" * level + "☆" * (3 - level)
                st.markdown(f"**{concept.title()}:** {stars}")
        else:
            st.info("Complete quizzes to see your progress!")

def run_active_quiz(quiz_data, quiz_id):
    """Run the currently active quiz"""
    st.subheader(f"🧠 {quiz_data['title']}")
    
    # Initialize quiz state
    if f"quiz_state_{quiz_id}" not in st.session_state:
        st.session_state[f"quiz_state_{quiz_id}"] = {
            'current_question': 0,
            'score': 0,
            'answers': [],
            'completed': False
        }
    
    quiz_state = st.session_state[f"quiz_state_{quiz_id}"]
    questions = quiz_data['questions']
    
    if not quiz_state['completed']:
        current_q = quiz_state['current_question']
        
        if current_q < len(questions):
            question_data = questions[current_q]
            
            # Progress indicator
            progress = (current_q + 1) / len(questions)
            st.progress(progress)
            st.markdown(f"**Question {current_q + 1} of {len(questions)}**")
            
            # Display question
            st.markdown(f"### {question_data['question']}")
            
            # Answer options
            answer = st.radio(
                "Choose your answer:",
                question_data['options'],
                key=f"q_{quiz_id}_{current_q}"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Submit Answer", type="primary"):
                    selected_index = question_data['options'].index(answer)
                    is_correct = selected_index == question_data['correct']
                    
                    quiz_state['answers'].append({
                        'question': question_data['question'],
                        'selected': answer,
                        'correct': is_correct,
                        'explanation': question_data['explanation']
                    })
                    
                    if is_correct:
                        quiz_state['score'] += 1
                        add_xp_enhanced(20, "Correct answer!", quiz_data['difficulty'], True)
                        st.success("✅ Correct!")
                        
                        # Update mastery
                        concept = quiz_data['concept']
                        current_mastery = st.session_state.mastery_levels.get(concept, 0)
                        st.session_state.mastery_levels[concept] = min(3, current_mastery + 1)
                        
                    else:
                        add_xp_enhanced(5, "Keep learning!", quiz_data['difficulty'], False)
                        st.error("❌ Incorrect")
                        correct_answer = question_data['options'][question_data['correct']]
                        st.info(f"The correct answer was: **{correct_answer}**")
                    
                    st.info(f"💡 **Explanation:** {question_data['explanation']}")
                    quiz_state['current_question'] += 1
                    
                    time.sleep(3)
                    st.experimental_rerun()
            
            with col2:
                # Hint system (if user has hint boosters)
                if st.session_state.shop_inventory.get('hint_booster', 0) > 0:
                    if st.button("💡 Use Hint"):
                        st.session_state.shop_inventory['hint_booster'] -= 1
                        concept = quiz_data['concept']
                        hint = QuantumTutor.adaptive_hint(concept, st.session_state.mastery_levels.get(concept, 0))
                        st.info(f"🔍 **Hint:** {hint}")
            
            with col3:
                if st.button("❌ Quit Quiz"):
                    del st.session_state[f"quiz_state_{quiz_id}"]
                    del st.session_state.current_quiz
                    st.experimental_rerun()
        
        else:
            quiz_state['completed'] = True
            st.experimental_rerun()
    
    else:
        # Quiz completed - show results
        st.success("🎉 Quiz Completed!")
        
        score = quiz_state['score']
        total = len(questions)
        percentage = (score / total) * 100
        
        # Store score in session state
        st.session_state.quiz_scores[quiz_id] = percentage
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage:.1f}%")
        with col3:
            if percentage >= 90:
                grade = "A+"
                st.metric("Grade", grade, delta="Excellent!")
            elif percentage >= 80:
                grade = "A"
                st.metric("Grade", grade, delta="Great!")
            elif percentage >= 70:
                grade = "B"
                st.metric("Grade", grade, delta="Good")
            elif percentage >= 60:
                grade = "C"
                st.metric("Grade", grade, delta="Passing")
            else:
                grade = "F"
                st.metric("Grade", grade, delta="Study more")
        
        # Performance feedback
        if percentage >= 80:
            st.success("🌟 Excellent work! You've mastered this concept!")
            if f"{quiz_id}_mastery" not in st.session_state.achievements:
                st.session_state.achievements.append(f"{quiz_data['title']} Mastery")
                st.balloons()
        elif percentage >= 60:
            st.info("👍 Good job! Consider reviewing the material for even better understanding.")
        else:
            st.warning("📚 Keep studying! Review the concept tutorial and try again.")
        
        # Detailed feedback
        if st.expander("📊 Detailed Results"):
            for i, answer in enumerate(quiz_state['answers']):
                status = "✅" if answer['correct'] else "❌"
                st.markdown(f"**Q{i+1}:** {answer['question']}")
                st.markdown(f"{status} Your answer: {answer['selected']}")
                st.markdown(f"💡 {answer['explanation']}")
                st.markdown("---")
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake Quiz"):
                del st.session_state[f"quiz_state_{quiz_id}"]
                st.experimental_rerun()
        
        with col2:
            if st.button("📚 Study This Topic"):
                del st.session_state.current_quiz
                st.experimental_rerun()

# ----------------------- Main App Entry Point ----------------------------

if __name__ == '__main__':
    main_app()index = question_data['options'].index(answer)
                    is_correct = selected_index == question_data['correct']
                    
                    quiz_state['answers'].append({
                        'question': question_data['question'],
                        'selected': answer,
                        'correct': is_correct,
                        'explanation': question_data['explanation']
                    })
                    
                    if is_correct:
                        quiz_state['score'] += 1
                        add_xp_enhanced(20, "Correct answer!", selected_quiz['difficulty'], True)
                        st.success("✅ Correct!")
                        st.info(f"💡 {question_data['explanation']}")
                        
                        # Update mastery level
                        concept = selected_quiz['concept']
                        st.session_state.mastery_levels[concept] = min(3, st.session_state.mastery_levels[concept] + 1)
                        
                    else:
                        add_xp_enhanced(5, "Keep learning!", selected_quiz['difficulty'], False)
                        st.error("❌ Incorrect")
                        st.info(f"💡 {question_data['explanation']}")
                        correct_answer = question_data['options'][question_data['correct']]
                        st.info(f"The correct answer was: **{correct_answer}**")
                    
                    quiz_state['current_question'] += 1
                    time.sleep(2)
                    st.experimental_rerun()
            
            with col2:
                if st.button("Get Hint"):
                    concept = selected_quiz['concept']
                    hint = QuantumTutor.adaptive_hint(concept, st.session_state.mastery_levels.get(concept, 0))
                    st.info(f"💡 **Hint:** {hint}")
        
        else:
            quiz_state['completed'] = True
            st.experimental_rerun()
    
    else:
        # Quiz completed - show results
        st.success("🎉 Quiz Completed!")
        
        score = quiz_state['score']
        total = len(questions)
        percentage = (score / total) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", f"{score}/{total}")
        with col2:
            st.metric("Percentage", f"{percentage:.1f}%")
        with col3:
            grade = "A+" if percentage >= 90 else "A" if percentage >= 80 else "B" if percentage >= 70 else "C" if percentage >= 60 else "F"
            st.metric("Grade", grade)
        
        # Performance analysis
        if percentage >= 80:
            st.success("🌟 Excellent work! You've mastered this concept!")
            if f"{quiz_name}_mastery" not in st.session_state.achievements:
                st.session_state.achievements.append(f"{quiz_name.replace('_', ' ').title()} Mastery")
        elif percentage >= 60:
            st.info("👍 Good job! Consider reviewing the material for even better understanding.")
        else:
            st.warning("📚 Keep studying! Review the concept tutorial and try again.")
        
        # Detailed feedback
        if st.expander("📊 Detailed Results"):
            for i, answer in enumerate(quiz_state['answers']):
                status = "✅" if answer['correct'] else "❌"
                st.markdown(f"**Q{i+1}:** {answer['question']}")
                st.markdown(f"{status} Your answer: {answer['selected']}")
                st.markdown(f"💡 {answer['explanation']}")
                st.markdown("---")
        
        # Reset quiz option
        if st.button("🔄 Retake Quiz"):
            del st.session_state[f"quiz_{quiz_name}"]
            st.experimental_rerun()

# ----------------------- Main App Structure ----------------------------

def main_app():
    st.title("⚛️ MathCraft: Quantum Quest Enhanced")
    st.caption("A comprehensive, story-driven introduction to quantum mechanics with AI tutoring — Built by Xavier Honablue M.Ed for CognitiveCloud.ai")

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
                "🌊 Interference Sandbox",
                "🎯 Stern–Gerlach Lab",
                "🔗 Entanglement Workshop",
                "📊 Quantum Simulator",
                "🧠 Enhanced Quiz Center",
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
    elif page == "🌊 Interference Sandbox":
        interference_sandbox()
    elif page == "🎯 Stern–Gerlach Lab":
        stern_gerlach_lab()
    elif page == "🔗 Entanglement Workshop":
        entanglement_workshop()
    elif page == "📊 Quantum Simulator":
        quantum_simulator()
    elif page == "🧠 Enhanced Quiz Center":
        enhanced_quiz_center()
    elif page == "🎮 Quantum Challenges":
        quantum_challenges()
    elif page == "💎 Quantum Shop":
        quantum_shop()
    elif page == "🚀 Future Quest Modules":
        future_quest_modules()
    elif page == "📚 Resources":
        resources_page()

# ----------------------- Enhanced Story Mode ----------------------------

def story_mode_enhanced():
    st.header("Episode 1: The Mysterious Q-Box Enhanced")
    
    # Progress tracking
    if 'story_progress' not in st.session_state:
        st.session_state.story_progress = {
            'episode_1_complete': False,
            'hadamard_mastered': False,
            'first_measurement': False
        }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 📖 The Enhanced Discovery
        
        In the dusty attic of an old physics laboratory, you discover a peculiar device labeled **Q-BOX 2.0**. 
        This isn't just any quantum device - it's equipped with an AI tutor and advanced quantum mechanics!
        
        A glowing holographic note appears:
        
        > *"Welcome, quantum explorer! This enhanced Q-Box will guide you through the mysteries of quantum mechanics.
        > Complete challenges, earn quantum coins, and unlock the deepest secrets of the quantum realm.
        > Your AI tutor is standing by to help!"*
        
        Your enhanced quantum journey begins now! 🚀
        """)
        
        # Adaptive difficulty based on user level
        if st.session_state.current_level >= 3:
            st.info("🎓 **Advanced Mode Unlocked:** More complex states and challenges available!")
            target_state = st.selectbox(
                "Choose your target state:",
                ["Equal Superposition |+⟩", "Minus State |-⟩", "Complex Phase State", "Custom Challenge"]
            )
        else:
            target_state = "Equal Superposition |+⟩"
        
        st.markdown(f"### 🎯 Current Mission: Create {target_state}")
        
        if target_state == "Equal Superposition |+⟩":
            st.info("Target: Prepare the state (|0⟩ + |1⟩)/√2 with equal probability of measuring 0 or 1")
            target_theta = np.pi/2
            target_phi = 0
        elif target_state == "Minus State |-⟩":
            st.info("Target: Prepare the state (|0⟩ - |1⟩)/√2")
            target_theta = np.pi/2
            target_phi = np.pi
        elif target_state == "Complex Phase State":
            st.info("Target: Prepare the state (|0⟩ + i|1⟩)/√2")
            target_theta = np.pi/2
            target_phi = np.pi/2
        else:
            st.info("Target: Create any superposition state you choose!")
            target_theta = None
            target_phi = None
        
        # Enhanced controls with hints
        col_theta, col_phi = st.columns(2)
        with col_theta:
            theta = st.slider("🔄 Rotation around Y-axis (θ)", 0.0, float(np.pi), value=float(np.pi/2), step=0.01)
            if st.button("💡 Theta Hint"):
                st.info("θ controls the 'tilt' of your state vector. π/2 creates maximum superposition!")
        
        with col_phi:
            phi = st.slider("🌀 Phase rotation around Z-axis (φ)", 0.0, float(2*np.pi), value=0.0, step=0.01)
            if st.button("💡 Phi Hint"):
                st.info("φ adds a phase. Try π for minus states, π/2 for imaginary phases!")
        
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
        
        # Smart measurement suggestions
        trials = st.number_input("Number of measurements", min_value=10, max_value=1000, value=100, step=10)
        
        if st.button("🎲 Run Quantum Measurements", type="primary"):
            with st.spinner("Measuring quantum state..."):
                time.sleep(0.5)
                results = [sample_measure(state) for _ in range(int(trials))]
                p0_observed = results.count("0") / len(results)
                p1_observed = 1 - p0_observed
                
                st.success(f"📊 Results: |0⟩ → {p0_observed:.3f}, |1⟩ → {p1_observed:.3f}")
                
                # Enhanced feedback system
                success = False
                if target_state == "Equal Superposition |+⟩":
                    if abs(p0_observed - 0.5) < 0.1 and abs(p1_observed - 0.5) < 0.1:
                        success = True
                elif target_state == "Minus State |-⟩":
                    if abs(p0_observed - 0.5) < 0.1 and abs(p1_observed - 0.5) < 0.1:
                        success = True
                elif target_state == "Complex Phase State":
                    if abs(p0_observed - 0.5) < 0.1 and abs(p1_observed - 0.5) < 0.1:
                        success = True
                else:
                    success = True  # Custom challenge always succeeds
                
                if success:
                    difficulty = 'easy' if target_state == "Equal Superposition |+⟩" else 'medium'
                    add_xp_enhanced(25, f"Successfully created {target_state}!", difficulty, True)
                    
                    if target_state == "Equal Superposition |+⟩" and "Hadamard Master" not in st.session_state.achievements:
                        st.session_state.achievements.append("Hadamard Master")
                        st.balloons()
                    elif target_state == "Minus State |-⟩" and "Phase Master" not in st.session_state.achievements:
                        st.session_state.achievements.append("Phase Master")
                    elif target_state == "Complex Phase State" and "Complex State Master" not in st.session_state.achievements:
                        st.session_state.achievements.append("Complex State Master")
                        
                else:
                    add_xp_enhanced(8, "Good attempt! Check the AI Tutor for guidance.", 'easy', False)
    
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
        
        # State analysis
        st.markdown("### 📊 State Analysis")
        if abs(probs['0'] - 0.5) < 0.01:
            st.success("Perfect superposition achieved!")
        elif abs(probs['0'] - 1) < 0.01:
            st.info("Pure |0⟩ state")
        elif abs(probs['1'] - 1) < 0.01:
            st.info("Pure |1⟩ state")
        else:
            st.warning("Mixed superposition state")

# ----------------------- AI Quantum Tutor ----------------------------

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
        if st.button("Go to Basic Quiz"):
            st.experimental_set_query_params(page="quiz")
    elif mastery == 1:
        st.info("Try intermediate challenges and explore the advanced lab features!")
    elif mastery == 2:
        st.info("You're almost there! Tackle the hardest challenges to achieve mastery!")
    else:
        st.success("🌟 You've mastered this concept! Help others or explore advanced topics!")
    
    # Study plan generator
    st.subheader("📋 Personalized Study Plan")
    
    weak_concepts = [concept for concept, level in st.session_state.mastery_levels.items() if level < 2]
    strong_concepts = [concept for concept, level in st.session_state.mastery_levels.items() if level >= 2]
    
    if weak_concepts:
        st.markdown("**Focus on these concepts:**")
        for concept in weak_concepts:
            level = st.session_state.mastery_levels[concept]
            st.markdown(f"• {concept.title()}: {'⭐' * level}{'☆' * (3 - level)}")
    
    if strong_concepts:
        st.markdown("**You've mastered:**")
        for concept in strong_concepts:
            st.markdown(f"• {concept.title()}: ⭐⭐⭐")

# ----------------------- Enhanced Lab Modules ----------------------------

def superposition_lab_enhanced():
    st.header("🧪 Enhanced Superposition Laboratory")
    st.markdown("Craft custom quantum states with AI guidance and advanced features")
    
    # Lab mode selection
    lab_mode = st.radio(
        "Choose lab mode:",
        ["🌱 Guided Mode", "🔬 Free Exploration", "🎯 Challenge Mode"],
        horizontal=True
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if lab_mode == "🌱 Guided Mode":
            st.subheader("AI-Guided State Design")
            st.info("Follow the AI tutor's guidance to create specific quantum states!")
            
            target = st.selectbox(
                "Choose a target state to create:",
                ["Equal superposition", "75% |0⟩ state", "Complex phase state", "Custom state"]
            )
            
            if target == "Equal superposition":
                st.markdown("**Goal:** Create |+⟩ = (|0⟩ + |1⟩)/√2")
                st.markdown("**Hint:** Set both amplitudes to 1/√2 ≈ 0.707")
            elif target == "75% |0⟩ state":
                st.markdown("**Goal:** Create a state with 75% probability of measuring |0⟩")
                st.markdown("**Hint:** |α|² = 0.75, so α ≈ 0.866")
            elif target == "Complex phase state":
                st.markdown("**Goal:** Create (|0⟩ + i|1⟩)/√2")
                st.markdown("**Hint:** Use imaginary amplitude for |1⟩")
        
        elif lab_mode == "🎯 Challenge Mode":
            st.subheader("Superposition Challenges")
            
            challenges = [
                {"name": "Perfect Balance", "description": "Create exactly 50/50 probability", "target_p0": 0.5},
                {"name": "Golden Ratio", "description": "P(|0⟩) = 0.618 (golden ratio)", "target_p0": 0.618},
                {"name": "Quantum Third", "description": "P(|0⟩) = 1/3", "target_p0": 0.333}
            ]
            
            selected_challenge = st.selectbox("Choose a challenge:", challenges, format_func=lambda x: x['name'])
            st.markdown(f"**Challenge:** {selected_challenge['description']}")
        
        # State designer (common to all modes)
        st.subheader("State Designer")
        
        if lab_mode != "🎯 Challenge Mode":
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
        else:
            # Challenge mode - let user try to hit target
            magnitude_0 = st.slider("Magnitude of α (for |0⟩)", 0.0, 1.0, 0.707, 0.001)
            phase_0 = st.slider("Phase of α (radians)", 0.0, 2*np.pi, 0.0, 0.01)
            
            # Calculate β to maintain normalization
            magnitude_1 = np.sqrt(1 - magnitude_0**2)
            phase_1 = st.slider("Phase of β (radians)", 0.0, 2*np.pi, 0.0, 0.01)
            
            a_real = magnitude_0 * np.cos(phase_0)
            a_imag = magnitude_0 * np.sin(phase_0)
            b_real = magnitude_1 * np.cos(phase_1)
            b_imag = magnitude_1 * np.sin(phase_1)
        
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
        
        # Challenge checking
        if lab_mode == "🎯 Challenge Mode":
            target_prob = selected_challenge['target_p0']
            error = abs(probs['0'] - target_prob)
            
            if error < 0.01:
                st.success("🎉 Challenge completed! Perfect accuracy!")
                add_xp_enhanced(30, f"Completed {selected_challenge['name']} challenge!", 'medium', True)
            elif error < 0.05:
                st.info(f"Close! Error: {error:.3f}")
            else:
                st.warning(f"Keep trying! Error: {error:.3f}")
        
        # Enhanced measurement system
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
        
        # State information panel
        st.subheader("📊 State Analysis")
        st.markdown(f"**Bloch coordinates:** ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Classify the state
        if abs(z - 1) < 0.01:
            st.info("🔵 Pure |0⟩ state")
        elif abs(z + 1) < 0.01:
            st.info("🔴 Pure |1⟩ state")
        elif abs(z) < 0.01:
            st.info("🟡 Equatorial state (maximum superposition)")
        else:
            st.info("🟣 General superposition state")

def quantum_gates_workshop_enhanced():
    st.header("⚙️ Enhanced Quantum Gates Workshop")
    st.markdown("Master quantum gates with interactive tutorials and advanced challenges")
    
    # Workshop mode
    workshop_mode = st.radio(
        "Choose workshop mode:",
        ["📚 Tutorial Mode", "🔧 Circuit Builder", "🏆 Gate Challenges"],
        horizontal=True
    )
    
    if workshop_mode == "📚 Tutorial Mode":
        st.subheader("Interactive Gate Tutorial")
        
        gate_tutorial = st.selectbox(
            "Select a gate to learn:",
            ["Pauli-X (NOT)", "Hadamard", "Pauli-Z", "Rotation Gates", "CNOT"]
        )
        
        if gate_tutorial == "Pauli-X (NOT)":
            st.markdown("""
            ### 🔄 Pauli-X Gate (Quantum NOT)
            
            The X gate flips the qubit state: |0⟩ → |1⟩ and |1⟩ → |0⟩
            
            **Matrix representation:**
            ```
            X = [0  1]
                [1  0]
            ```
            """)
            
            st.markdown("**Try it yourself:**")
            initial = st.selectbox("Starting state:", ["|0⟩", "|1⟩", "Superposition"])
            
            if initial == "|0⟩":
                state = ket0()
                st.code("Initial: |0⟩")
            elif initial == "|1⟩":
                state = ket1()
                st.code("Initial: |1⟩")
            else:
                state = apply(H, ket0())
                st.code("Initial: |+⟩ = (|0⟩ + |1⟩)/√2")
            
            if st.button("Apply X Gate"):
                final_state = apply(X, state)
                st.code(f"Final: {cfmt(final_state[0,0])}|0⟩ + {cfmt(final_state[1,0])}|1⟩")
                add_xp_enhanced(8, "Applied X gate successfully!", 'easy', True)
    
    elif workshop_mode == "🏆 Gate Challenges":
        st.subheader("Gate Mastery Challenges")
        
        challenges = [
            {
                "name": "NOT Twice",
                "description": "Apply X gate twice to |0⟩. What do you get?",
                "initial_state": "|0⟩",
                "gates": ["X", "X"],
                "expected": "|0⟩"
            },
            {
                "name": "Create |1⟩",
                "description": "Transform |0⟩ to |1⟩ using any single gate",
                "initial_state": "|0⟩",
                "gates": ["?"],
                "expected": "|1⟩"
            },
            {
                "name": "Superposition Creation",
                "description": "Create equal superposition from |0⟩",
                "initial_state": "|0⟩",
                "gates": ["?"],
                "expected": "|+⟩"
            }
        ]
        
        selected_
