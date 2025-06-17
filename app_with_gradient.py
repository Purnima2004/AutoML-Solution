import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from preprocessing_hackathon import preprocess_data

# ==============================================
# ANIMATED GRADIENT BACKGROUND HTML COMPONENT
# ==============================================
GRADIENT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Animated Gradient Background</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Arial', sans-serif;
            overflow-x: hidden;
            background: #000;
        }

        .animated-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #0f0a1e;
            overflow: hidden;
            z-index: -1;
        }

        .animated-bg::before,
        .animated-bg::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(ellipse 800px 600px at var(--x, 50%) var(--y, 50%), 
                rgba(139, 92, 246, 0.8) 0%, 
                rgba(109, 40, 217, 0.6) 25%, 
                rgba(76, 29, 149, 0.4) 50%, 
                rgba(30, 27, 75, 0.2) 75%, 
                transparent 100%);
            animation: jellyfishMove 12s ease-in-out infinite;
            opacity: 0.9;
        }

        .animated-bg::after {
            background: radial-gradient(ellipse 600px 800px at var(--x2, 70%) var(--y2, 30%), 
                rgba(168, 85, 247, 0.6) 0%, 
                rgba(147, 51, 234, 0.5) 25%, 
                rgba(126, 34, 206, 0.3) 50%, 
                rgba(88, 28, 135, 0.2) 75%, 
                transparent 100%);
            animation: jellyfishMove2 15s ease-in-out infinite reverse;
            opacity: 0.7;
        }

        .jellyfish-layer {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: conic-gradient(from 0deg at var(--cx, 40%) var(--cy, 60%), 
                rgba(139, 92, 246, 0.3) 0deg, 
                rgba(168, 85, 247, 0.2) 90deg, 
                rgba(147, 51, 234, 0.4) 180deg, 
                rgba(109, 40, 217, 0.2) 270deg, 
                rgba(139, 92, 246, 0.3) 360deg);
            animation: jellyfishPulse 20s ease-in-out infinite;
            opacity: 0.6;
            filter: blur(60px);
        }

        @keyframes jellyfishMove {
            0%, 100% {
                --x: 30%;
                --y: 40%;
                transform: scale(1) rotate(0deg);
                filter: blur(0px) hue-rotate(0deg);
            }
            25% {
                --x: 70%;
                --y: 20%;
                transform: scale(1.2) rotate(10deg);
                filter: blur(5px) hue-rotate(30deg);
            }
            50% {
                --x: 80%;
                --y: 70%;
                transform: scale(0.8) rotate(-5deg);
                filter: blur(8px) hue-rotate(60deg);
            }
            75% {
                --x: 20%;
                --y: 80%;
                transform: scale(1.1) rotate(15deg);
                filter: blur(3px) hue-rotate(90deg);
            }
        }

        @keyframes jellyfishMove2 {
            0%, 100% {
                --x2: 60%;
                --y2: 30%;
                transform: scale(0.9) rotate(0deg);
                filter: blur(10px) hue-rotate(0deg);
            }
            33% {
                --x2: 20%;
                --y2: 60%;
                transform: scale(1.3) rotate(-8deg);
                filter: blur(15px) hue-rotate(-20deg);
            }
            66% {
                --x2: 90%;
                --y2: 40%;
                transform: scale(0.7) rotate(12deg);
                filter: blur(5px) hue-rotate(-40deg);
            }
        }

        @keyframes jellyfishPulse {
            0%, 100% {
                --cx: 40%;
                --cy: 60%;
                transform: scale(1) rotate(0deg);
                opacity: 0.3;
            }
            20% {
                --cx: 60%;
                --cy: 30%;
                transform: scale(1.4) rotate(45deg);
                opacity: 0.6;
            }
            40% {
                --cx: 80%;
                --cy: 70%;
                transform: scale(0.8) rotate(90deg);
                opacity: 0.4;
            }
            60% {
                --cx: 30%;
                --cy: 80%;
                transform: scale(1.2) rotate(135deg);
                opacity: 0.7;
            }
            80% {
                --cx: 70%;
                --cy: 20%;
                transform: scale(0.9) rotate(180deg);
                opacity: 0.5;
            }
        }

        .floating-elements {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .jellyfish-tentacles {
            position: absolute;
            width: 100%;
            height: 100%;
        }

        .tentacle {
            position: absolute;
            width: 2px;
            height: 200px;
            background: linear-gradient(to bottom, rgba(139, 92, 246, 0.6), transparent);
            transform-origin: top center;
            animation: tentacleWave 8s ease-in-out infinite;
        }

        .tentacle:nth-child(1) {
            left: 20%;
            top: 30%;
            animation-delay: 0s;
            height: 150px;
        }

        .tentacle:nth-child(2) {
            left: 35%;
            top: 45%;
            animation-delay: -2s;
            height: 180px;
        }

        .tentacle:nth-child(3) {
            left: 60%;
            top: 25%;
            animation-delay: -4s;
            height: 120px;
        }

        .tentacle:nth-child(4) {
            left: 75%;
            top: 40%;
            animation-delay: -6s;
            height: 160px;
        }

        .tentacle:nth-child(5) {
            left: 45%;
            top: 60%;
            animation-delay: -1s;
            height: 140px;
        }

        @keyframes tentacleWave {
            0%, 100% {
                transform: rotate(0deg) scaleY(1);
                opacity: 0.6;
            }
            25% {
                transform: rotate(15deg) scaleY(1.2);
                opacity: 0.8;
            }
            50% {
                transform: rotate(-10deg) scaleY(0.8);
                opacity: 0.4;
            }
            75% {
                transform: rotate(20deg) scaleY(1.1);
                opacity: 0.9;
            }
        }

        .floating-element {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            animation: organicFloat 18s infinite ease-in-out;
            filter: blur(1px);
        }

        .floating-element:nth-child(1) {
            width: 120px;
            height: 80px;
            top: 15%;
            left: 8%;
            border-radius: 50% 70% 40% 60%;
            animation-delay: 0s;
            background: rgba(139, 92, 246, 0.15);
        }

        .floating-element:nth-child(2) {
            width: 90px;
            height: 90px;
            top: 55%;
            right: 12%;
            border-radius: 60% 40% 70% 30%;
            animation-delay: -6s;
            background: rgba(168, 85, 247, 0.12);
        }

        .floating-element:nth-child(3) {
            width: 160px;
            height: 120px;
            top: 35%;
            left: 55%;
            border-radius: 40% 60% 50% 80%;
            animation-delay: -12s;
            background: rgba(147, 51, 234, 0.1);
        }

        .floating-element:nth-child(4) {
            width: 70px;
            height: 50px;
            top: 75%;
            left: 25%;
            border-radius: 80% 20% 60% 40%;
            animation-delay: -9s;
            background: rgba(255, 255, 255, 0.06);
        }

        .floating-element:nth-child(5) {
            width: 140px;
            height: 100px;
            top: 8%;
            right: 20%;
            border-radius: 30% 70% 80% 20%;
            animation-delay: -15s;
            background: rgba(126, 34, 206, 0.08);
        }

        @keyframes organicFloat {
            0%, 100% {
                transform: translateY(0) translateX(0) rotate(0deg) scale(1);
                border-radius: 50% 70% 40% 60%;
                opacity: 0.3;
            }
            16% {
                transform: translateY(-30px) translateX(15px) rotate(45deg) scale(1.1);
                border-radius: 60% 40% 80% 20%;
                opacity: 0.5;
            }
            33% {
                transform: translateY(-20px) translateX(-10px) rotate(90deg) scale(0.9);
                border-radius: 40% 80% 30% 70%;
                opacity: 0.4;
            }
            50% {
                transform: translateY(-50px) translateX(5px) rotate(135deg) scale(1.2);
                border-radius: 70% 30% 60% 40%;
                opacity: 0.6;
            }
            66% {
                transform: translateY(-15px) translateX(-20px) rotate(180deg) scale(0.8);
                border-radius: 30% 60% 70% 40%;
                opacity: 0.3;
            }
            83% {
                transform: translateY(-35px) translateX(10px) rotate(225deg) scale(1.05);
                border-radius: 80% 20% 50% 60%;
                opacity: 0.7;
            }
        }

        /* Particle effect */
        .particle {
            position: absolute;
            width: 2px;
            height: 2px;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 50%;
            animation: sparkle 3s infinite;
        }

        @keyframes sparkle {
            0%, 100% {
                opacity: 0;
                transform: scale(0);
            }
            50% {
                opacity: 1;
                transform: scale(1);
            }
        }
    </style>
</head>
<body>
    <div class="animated-bg">
        <div class="jellyfish-layer"></div>
        
        <div class="floating-elements">
            <div class="floating-element"></div>
            <div class="floating-element"></div>
            <div class="floating-element"></div>
            <div class="floating-element"></div>
            <div class="floating-element"></div>
        </div>

        <div class="jellyfish-tentacles">
            <div class="tentacle"></div>
            <div class="tentacle"></div>
            <div class="tentacle"></div>
            <div class="tentacle"></div>
            <div class="tentacle"></div>
        </div>
    </div>

    <script>
        // Add random particles
        function createParticles() {
            const container = document.querySelector('.floating-elements');
            
            for (let i = 0; i < 20; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.top = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 3 + 's';
                container.appendChild(particle);
            }
        }

        // Initialize particles
        createParticles();

        // Enhanced mouse movement with jellyfish effect
        document.addEventListener('mousemove', (e) => {
            const elements = document.querySelectorAll('.floating-element');
            const tentacles = document.querySelectorAll('.tentacle');
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;

            // Floating elements respond to mouse
            elements.forEach((element, index) => {
                const speed = (index + 1) * 0.3;
                const xPos = (x - 0.5) * speed * 15;
                const yPos = (y - 0.5) * speed * 15;
                
                element.style.transform += ` translate(${xPos}px, ${yPos}px)`;
            });

            // Tentacles react to mouse movement
            tentacles.forEach((tentacle, index) => {
                const speed = (index + 1) * 0.2;
                const rotation = (x - 0.5) * speed * 10;
                tentacle.style.transform += ` rotate(${rotation}deg)`;
            });

            // Update CSS custom properties for gradient movement
            document.documentElement.style.setProperty('--mouse-x', `${x * 100}%`);
            document.documentElement.style.setProperty('--mouse-y', `${y * 100}%`);
        });

        // Add breathing effect to the main gradient
        let breathePhase = 0;
        setInterval(() => {
            breathePhase += 0.02;
            const breatheScale = 1 + Math.sin(breathePhase) * 0.1;
            const jellyfishLayer = document.querySelector('.jellyfish-layer');
            if (jellyfishLayer) {
                jellyfishLayer.style.transform = `scale(${breatheScale})`;
            }
        }, 50);
    </script>
</body>
</html>
"""

# ==============================================
# STREAMLIT APP CONFIGURATION
# ==============================================
st.set_page_config(
    page_title="AutoML Solution",
    page_icon="✨",
    layout="wide"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# ==============================================
# MAIN APP LAYOUT
# ==============================================

# Render gradient background (fixed position)
st.components.v1.html(GRADIENT_HTML, height=0)

# Main content container with semi-transparent background
with st.container():
    st.markdown("""
    <style>
    .main-content {
        background-color: rgba(15, 10, 30, 0.85);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(5px);
    }
    </style>
    <div class="main-content">
    """, unsafe_allow_html=True)

    # Theme toggle button
    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([6, 1])
    with col2:
        current_theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
        theme_help = "Switch to dark mode" if st.session_state.theme == 'light' else "Switch to light mode"
        if st.button(current_theme_icon, help=theme_help, key="theme_toggle"):
            toggle_theme()
            st.experimental_rerun()

    # [REST OF YOUR EXISTING STREAMLIT CODE GOES HERE...]
    # ==============================================
    # [YOUR EXISTING AUTOML CODE FROM app.py]
    # ==============================================
    # (Include all your existing code starting from the theme colors definition
    # through to the end of the file, exactly as you had it)
    
    # Title and description with animation
    st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
    st.title("AutoML Solution")
    st.markdown("<h2 class='subtitle'>Just upload your dataset, select parameters, and train the models and see the Magic!</h2>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # [CONTINUE WITH ALL YOUR EXISTING CODE...]
    
    # Close the main content container
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================
# ADDITIONAL STYLING FOR BETTER INTEGRATION
# ==============================================
st.markdown("""
<style>
/* Make sure Streamlit elements are visible over gradient */
.stApp > div {
    background-color: transparent !important;
}

/* Adjust text colors for better contrast */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText {
    color: white !important;
    text-shadow: 0 0 5px rgba(0,0,0,0.5);
}

/* Style buttons to stand out */
.stButton>button {
    background-color: rgba(139, 92, 246, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    color: white !important;
}

/* Style file uploader */
.stFileUploader>div {
    background-color: rgba(255,255,255,0.1) !important;
    backdrop-filter: blur(5px);
}

/* Adjust dataframes */
.stDataFrame {
    background-color: rgba(30, 27, 75, 0.7) !important;
}
</style>
""", unsafe_allow_html=True)