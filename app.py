import streamlit as st

st.set_page_config(
    page_title="CognitiveCloud.ai Math Apps",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .app-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem;
        text-align: center;
        color: white;
        text-decoration: none;
        transition: transform 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .app-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .app-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .app-description {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .new-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 CognitiveCloud.ai App Launcher</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore Xavier Honablue\'s interactive math apps - Choose your mathematical adventure!</p>', unsafe_allow_html=True)

# App data, organized by subject
subjects = {
    "Mindset": [
        { "name": "🌱 Growth Mindset Explorer", "description": "Build confidence and resilience through math challenges", "url": "https://growth-mindset-ed.streamlit.app/", "icon": "🌱", "color": "#81C784" },
        { "name": "🌟 Positive Mindset Math", "description": "4th Grade Multiplication with growth mindset activities", "url": "https://positive.streamlit.app/", "icon": "🌟", "color": "#FFB6C1" }
    ],
    "Foundational & General Math": [
        { "name": "📜 Greek Letters", "description": "Explore the Greek alphabet's use in mathematics and science.", "url": "https://math-greeks.streamlit.app/", "icon": "📜", "color": "#8A2BE2" }
    ],
    "Early Math": [
        { "name": "➗ Division Dash", "description": "Practice division skills with interactive games and quizzes", "url": "https://division.streamlit.app/", "icon": "➗", "color": "#FFC0CB" },
        { "name": "🍕 Fractions (Pizza Cutter)", "description": "Learn fractions with interactive pizza slices", "url": "https://pizza-math.streamlit.app/", "icon": "🍕", "color": "#DDA0DD" },
        { "name": "✏️ Pencil Dashboard", "description": "4th grade math lesson with surface area", "url": "https://pencil-dashboard.streamlit.app/", "icon": "✏️", "color": "#82E0AA" },
        { "name": "⚖️ Ratios & Proportions", "description": "Explore the relationship between quantities and solve proportions.", "url": "https://mathcraft-ratios.streamlit.app/", "icon": "⚖️", "color": "#76D7C4" },
        { "name": "💯 Percents", "description": "Learn percents, percent change, and conversions", "url": "https://percents.streamlit.app/", "icon": "💯", "color": "#FF8C00" }
    ],
    "Statistics": [
        { "name": "🔎 Data Detectives: Mean, Median, Mode, and Range", "description": "An interactive lesson on mean, median, mode, and range.", "url": "https://data-stats.streamlit.app/", "icon": "🔎", "color": "#FFD700" },
        { "name": "🎯 Vision: See the Distribution", "description": "Explore normal distributions, real-world data, and global impact", "url": "https://vision-distribution.streamlit.app/", "icon": "🎯", "color": "#FAD02C" },
        { "name": "📊 Stochastics Math", "description": "Learn probability, random processes, and their applications", "url": "https://stochastics-math.streamlit.app/", "icon": "📊", "color": "#48C9B0" }
    ],
    "Algebra": [
        { "name": "📐 Algebra Rules", "description": "Master algebraic concepts and rules", "url": "https://mathcraft-algebrarules1.streamlit.app/", "icon": "📐", "color": "#FF6B6B" },
        { "name": "🚂 Train Motion", "description": "Algebra with motion problems", "url": "https://mathcraft-trainmotion.streamlit.app/", "icon": "🚂", "color": "#F9E79F" },
        { "name": "📈 Distribution Curves", "description": "Statistical distributions and probability", "url": "https://algebra1rules-q94clpmtxy8hzwvxs6jxwx.streamlit.app/", "icon": "📈", "color": "#FFEAA7" }
    ],
    "Geometry & Visual Math": [
        { "name": "🔴 Conic Sections", "description": "Visualize circles, ellipses, parabolas & hyperbolas", "url": "https://mathcraft-conicsections.streamlit.app/", "icon": "🔴", "color": "#45B7D1" },
        { "name": "🌀 Spiral Vision: Fibonacci & Golden Ratio", "description": "Discover nature's secret mathematical spiral patterns", "url": "https://fibonacci-ratio.streamlit.app/", "icon": "🌀", "color": "#E17055" },
        { "name": "🔺 Tessellations", "description": "Geometric pattern exploration", "url": "https://mathcraft-tesselations.streamlit.app/", "icon": "🔺", "color": "#D7BDE2" },
        { "name": "🎯 Trigonometry", "description": "Sine, cosine, tangent, and more", "url": "https://mathcraft-trigonometry.streamlit.app/", "icon": "🎯", "color": "#BB8FCE" }
    ],
    "Calculus & Advanced Topics": [
        { "name": "📐 Riemann Sums", "description": "Visualize the area under a curve by summing rectangles.", "url": "https://reimannsum.streamlit.app/", "icon": "📐", "color": "#6A5ACD" },
        { "name": "📉 Calculus: Inflection Explorer", "description": "Visualize concavity, inflection points, and second derivatives", "url": "https://calculus-inflections.streamlit.app/", "icon": "📉", "color": "#A29BFE" },
        { "name": "📊 Calculus", "description": "Explore derivatives, integrals, and limits", "url": "https://calculus.streamlit.app/", "icon": "📊", "color": "#4ECDC4" },
        { "name": "🔢 Discrete Structures", "description": "Logic, sets, and combinatorics", "url": "https://mathcraft-discretestructures.streamlit.app/", "icon": "🔢", "color": "#96CEB4" },
        { "name": "⚡ Irrational Numbers", "description": "Explore pi, e, and other irrational numbers", "url": "https://mathcraft-irrationalnumbers.streamlit.app/", "icon": "⚡", "color": "#F7DC6F" },
        { "name": "📊 Polynomial Zeros", "description": "Explore polynomial functions and find their zeros/roots", "url": "https://mathcraft-twistedcurves.streamlit.app/", "icon": "📊", "color": "#F8C471" }
    ],
    "Physics & Applied Math": [
        { "name": "💧 Bernoulli's Principle", "description": "Explore fluid dynamics and the inverse relationship between speed and pressure.", "url": "https://bernoulli-princial.streamlit.app/", "icon": "💧", "color": "#5DADE2" },
        { "name": "🧲 Magnetism & Math", "description": "Explore magnetic fields, equations, and math connections", "url": "https://mathofmagnetism.streamlit.app/", "icon": "🧲", "color": "#00B894" },
        { "name": "🔄 Particle Motion", "description": "Physics and motion visualizations", "url": "https://mathcraft-particlemotion.streamlit.app/", "icon": "🔄", "color": "#85C1E9" },
        { "name": "🛩️ Aviation Math", "description": "Flight-based math with angles, altitude, and time-distance problems", "url": "https://aviation-math.streamlit.app/", "icon": "🛩️", "color": "#AED6F1" },
        { "name": "⚛️ MathCraft: Quantum Quest", "description": "Interactive quantum mechanics with AI tutoring, gamification, and comprehensive learning modules", "url": "https://quantum-mechanics.streamlit.app/", "icon": "⚛️", "color": "#9B59B6", "new": True }
    ]
}

# Create a grid layout
cols_per_row = 3

for subject, app_list in subjects.items():
    st.markdown("---")
    st.markdown(f'<h2 style="font-size: 2rem; color: #1f77b4;">{subject}</h2>', unsafe_allow_html=True)
    rows = [app_list[i:i + cols_per_row] for i in range(0, len(app_list), cols_per_row)]
    
    for row in rows:
        cols = st.columns(len(row))
        for col, app in zip(cols, row):
            with col:
                # Add NEW badge for new apps
                new_badge = '<span class="new-badge">NEW!</span>' if app.get('new', False) else ''
                
                st.markdown(f"""
                <a href="{app['url']}" target="_blank" style="text-decoration: none;">
                    <div style="
                        background: linear-gradient(135deg, {app['color']}22 0%, {app['color']}44 100%);
                        border: 2px solid {app['color']};
                        padding: 1.5rem;
                        border-radius: 15px;
                        text-align: center;
                        margin: 0.5rem 0;
                        transition: all 0.3s ease;
                        position: relative;
                    ">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{app['icon']}</div>
                        <h3 style="color: #333; margin-bottom: 0.5rem; font-size: 1.5rem;">
                            {app['name'].split(' ', 1)[1]}{new_badge}
                        </h3>
                        <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem; min-height: 45px;">{app['description']}</p>
                        <div style="
                            background: {app['color']};
                            color: white;
                            padding: 0.75rem 1.5rem;
                            border-radius: 25px;
                            text-decoration: none;
                            font-weight: bold;
                            transition: all 0.3s ease;
                            display: inline-block;
                        ">Launch App</div>
                    </div>
                </a>
                """, unsafe_allow_html=True)

# Highlight the new Quantum Quest app
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #9B59B6, #3498DB);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin: 2rem 0;
    box-shadow: 0 10px 30px rgba(155, 89, 182, 0.3);
">
    <h2 style="margin-bottom: 1rem; font-size: 2.5rem;">🚀 Featured: NEW Quantum Quest!</h2>
    <p style="font-size: 1.2rem; margin-bottom: 1.5rem;">
        Dive into the fascinating world of quantum mechanics with our most advanced learning platform yet!
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1.5rem;">
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
            <div style="font-size: 2rem;">🎓</div>
            <div>AI Tutoring</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
            <div style="font-size: 2rem;">🎮</div>
            <div>Gamification</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
            <div style="font-size: 2rem;">🧠</div>
            <div>25+ Quizzes</div>
        </div>
        <div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px;">
            <div style="font-size: 2rem;">⚛️</div>
            <div>Quantum Coins</div>
        </div>
    </div>
    <a href="https://quantum-mechanics.streamlit.app/" target="_blank" style="
        background: white;
        color: #9B59B6;
        padding: 1rem 2rem;
        border-radius: 30px;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
        transition: all 0.3s ease;
    ">🚀 Launch Quantum Quest</a>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: #666;">
    <p>💡 <strong>Empowering Young Minds in STEAM</strong></p>
    <p>Innovative learning solutions for grades 4-12 | Created by Xavier Honablue</p>
    <p style="margin-top: 1rem; font-style: italic;">
        ⚛️ Now featuring quantum mechanics education with advanced AI tutoring and gamification!
    </p>
</div>
""", unsafe_allow_html=True)

# Updated Stats
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
total_apps = sum(len(apps) for apps in subjects.values())
with col1:
    st.metric("🎯 Total Apps", total_apps)
with col2:
    st.metric("👥 Grade Levels", "4-12")
with col3:
    st.metric("📚 Subject Areas", len(subjects))
with col4:
    st.metric("⚛️ New Features", "Quantum AI Tutor")
