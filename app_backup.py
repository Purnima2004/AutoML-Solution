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

# Import your preprocessing function
from preprocessing_hackathon import preprocess_data

# Must be the first Streamlit command
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

# Enhanced border colors with morphing gradients
light_mode_border = """
    background: linear-gradient(135deg, 
        hsl(330, 80%, 85%) 0%,
        hsl(320, 70%, 80%) 25%,
        hsl(310, 75%, 85%) 50%,
        hsl(300, 80%, 90%) 75%,
        hsl(290, 70%, 85%) 100%
    );
    background-size: 200% 200%;
    animation: lightModeShift 8s ease-in-out infinite;
"""

dark_mode_border = """
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        #FF69B4 0%,
        #8A2BE2 20%,
        #4B0082 40%,
        #9400D3 60%,
        #FFE4B5 80%,
        #FF69B4 100%
    );
    background-size: 300% 300%;
    animation: darkModeShift 6s ease-in-out infinite;
"""

if st.session_state.theme == 'light':
    bg_color = '#fff0f6'  # Light rose pink
    text_color = '#1a1a1a'
    gradient_css = """
        radial-gradient(
            circle at var(--mouse-x) var(--mouse-y),
            #ffe4ec 0%,
            #ffc1e3 40%,
            #ffb6c1 70%,
            #fff0f6 100%
        )
    """
else:
    bg_color = '#1a1a1a'
    text_color = '#ffffff'
    gradient_css = """
        radial-gradient(
            circle at var(--mouse-x) var(--mouse-y),
            #a259ff 0%,
            #6a1b9a 40%,
            #8a2be2 70%,
            #4b0082 100%
        )
    """


# Enhanced CSS with cursor tracking and morphing effects
enhanced_theme_css = f"""
/* Cursor tracking variables */
:root {{
    --mouse-x: 50%;
    --mouse-y: 50%;
    --bg-color: {bg_color};
    --text-color: {text_color};
}}

/* Base theme */
.stApp {{
    background-color: {bg_color};
    color: {text_color};
    transition: background-color 0.3s ease, color 0.3s ease;
    position: relative;
}}

/* Morphing background overlay */
.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        {'rgba(255, 182, 193, 0.05)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.08)'} 0%,
        {'rgba(255, 160, 200, 0.03)' if st.session_state.theme == 'light' else 'rgba(138, 43, 226, 0.06)'} 25%,
        {'rgba(255, 192, 203, 0.02)' if st.session_state.theme == 'light' else 'rgba(75, 0, 130, 0.04)'} 50%,
        {'rgba(240, 128, 128, 0.03)' if st.session_state.theme == 'light' else 'rgba(148, 0, 211, 0.06)'} 75%,
        {'rgba(255, 218, 185, 0.02)' if st.session_state.theme == 'light' else 'rgba(255, 228, 181, 0.04)'} 100%
    );
    pointer-events: none;
    z-index: -1;
    transition: all 0.3s ease;
}}

/* Enhanced main container with morphing border */
.main .block-container {{
    padding: 2rem;
    border-radius: 20px;
    position: relative;
    overflow: hidden;
    background: {bg_color};
    transition: all 0.3s ease;
}}

.main .block-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 20px;
    padding: 3px;
    {light_mode_border if st.session_state.theme == 'light' else dark_mode_border}
    z-index: -2;
}}

.main .block-container::after {{
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    right: 3px;
    bottom: 3px;
    background: {bg_color};
    border-radius: 17px;
    z-index: -1;
}}

/* Enhanced file uploader with cursor morphing */
.stFileUploader {{
    position: relative;
    border-radius: 15px !important;
    padding: 1rem !important;
    overflow: hidden;
    transition: all 0.3s ease;
}}

.stFileUploader::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 15px;
    padding: 2px;
    background: {'linear-gradient(45deg, #FFB6C1, #FFC0CB, #FFB6C1, #F0E68C)' if st.session_state.theme == 'light' else 'linear-gradient(45deg, #FF69B4, #8A2BE2, #4B0082, #9400D3, #FFE4B5)'};
    background-size: 400% 400%;
    animation: morphingBorder 6s ease-in-out infinite;
    z-index: -1;
}}

.stFileUploader::after {{
    content: '';
    position: absolute;
    top: 2px;
    left: 2px;
    right: 2px;
    bottom: 2px;
    background: {bg_color};
    border-radius: 13px;
    z-index: -1;
}}

/* Enhanced theme toggle button */
[data-testid="stButton"] button:has(div:contains("🌙")), 
[data-testid="stButton"] button:has(div:contains("☀️")) {{
    position: relative;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    overflow: hidden;
    background: transparent !important;
    border: none !important;
    animation: pulse 3s infinite;
}}

[data-testid="stButton"] button:has(div:contains("🌙"))::before,
[data-testid="stButton"] button:has(div:contains("☀️"))::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: {'conic-gradient(from 0deg, #FFB6C1, #FFC0CB, #FFB6C1, #F0E68C)' if st.session_state.theme == 'light' else 'conic-gradient(from 0deg, #FF69B4, #8A2BE2, #4B0082, #9400D3, #FFE4B5)'};
    animation: rotateMorph 10s linear infinite;
    z-index: -1;
}}

/* Morphing animations */
@keyframes lightModeShift {{
    0%, 100% {{ 
        background-position: 0% 50%;
        filter: hue-rotate(0deg);
    }}
    25% {{ 
        background-position: 50% 0%;
        filter: hue-rotate(15deg);
    }}
    50% {{ 
        background-position: 100% 50%;
        filter: hue-rotate(0deg);
    }}
    75% {{ 
        background-position: 50% 100%;
        filter: hue-rotate(-15deg);
    }}
}}

@keyframes darkModeShift {{
    0%, 100% {{ 
        background-position: 0% 50%;
        filter: hue-rotate(0deg) saturate(1.2);
    }}
    25% {{ 
        background-position: 50% 0%;
        filter: hue-rotate(45deg) saturate(1.4);
    }}
    50% {{ 
        background-position: 100% 50%;
        filter: hue-rotate(90deg) saturate(1.6);
    }}
    75% {{ 
        background-position: 50% 100%;
        filter: hue-rotate(45deg) saturate(1.4);
    }}
}}

@keyframes morphingBorder {{
    0%, 100% {{ 
        background-position: 0% 50%;
        transform: rotate(0deg);
    }}
    25% {{ 
        background-position: 100% 0%;
        transform: rotate(90deg);
    }}
    50% {{ 
        background-position: 100% 100%;
        transform: rotate(180deg);
    }}
    75% {{ 
        background-position: 0% 100%;
        transform: rotate(270deg);
    }}
}}

@keyframes rotateMorph {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}

@keyframes pulse {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.05); }}
}}

/* Enhanced button styling with hover morphing */
button {{
    position: relative;
    border: 2px solid transparent !important;
    color: {text_color} !important;
    background: {bg_color} !important;
    border-radius: 8px !important;
    overflow: hidden;
    transition: all 0.3s ease;
}}

button::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: {'linear-gradient(45deg, #FFB6C1, #FFC0CB)' if st.session_state.theme == 'light' else 'linear-gradient(45deg, #FF69B4, #8A2BE2)'};
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: -1;
}}

button:hover::before {{
    opacity: 0.2;
}}

button:hover {{
    transform: scale(1.02) !important;
    box-shadow: 0 0 20px {'rgba(255, 182, 193, 0.4)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.4)'} !important;
}}

/* Enhanced metric cards with morphing effect */
[data-testid="stMetric"] {{
    position: relative;
    background: transparent !important;
    border-radius: 12px;
    padding: 20px !important;
    overflow: hidden;
    transition: all 0.3s ease;
}}

[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        {'rgba(255, 182, 193, 0.1)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.1)'} 0%,
        {'rgba(255, 160, 200, 0.05)' if st.session_state.theme == 'light' else 'rgba(138, 43, 226, 0.05)'} 100%
    );
    border-radius: 12px;
    z-index: -1;
    transition: all 0.3s ease;
}}

[data-testid="stMetric"]:hover::before {{
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        {'rgba(255, 182, 193, 0.2)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.2)'} 0%,
        {'rgba(255, 160, 200, 0.1)' if st.session_state.theme == 'light' else 'rgba(138, 43, 226, 0.1)'} 100%
    );
}}

/* All other existing styles remain the same */
[data-testid="stSidebar"] {{
    background-color: {bg_color};
    border-right: 1px solid;
    border-image: {'linear-gradient(135deg, #FFB6C1, #FFC0CB)' if st.session_state.theme == 'light' else 'linear-gradient(135deg, #FF69B4, #8A2BE2)'} 1;
}}

/* Text styling */
.stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, span, div {{
    color: {text_color} !important;
}}

/* Widget labels */
.stSelectbox label, .stMultiselect label, .stSlider label, .stNumberInput label, .stTextInput label, .stTextArea label {{
    color: {text_color} !important;
}}

/* Input fields styling */
.stTextInput input, .stNumberInput input, .stTextArea textarea {{
    background-color: {'#f9f9f9' if st.session_state.theme == 'light' else '#2b2b2b'} !important;
    color: {text_color} !important;
    border: 1px solid {'#ddd' if st.session_state.theme == 'light' else '#444'} !important;
}}

/* Responsive design */
@media (max-width: 768px) {{
    .stApp::before {{
        background-size: 200% 200%;
        animation: mobileMorph 8s ease-in-out infinite;
    }}
}}

@keyframes mobileMorph {{
    0%, 100% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
}}
"""

# JavaScript for cursor tracking
cursor_tracking_js = """
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Cursor tracking for morphing gradients
    document.addEventListener('mousemove', (e) => {
        const mouseX = (e.clientX / window.innerWidth) * 100;
        const mouseY = (e.clientY / window.innerHeight) * 100;
        
        document.documentElement.style.setProperty('--mouse-x', `${mouseX}%`);
        document.documentElement.style.setProperty('--mouse-y', `${mouseY}%`);
    });

    // Enhanced touch support for mobile
    document.addEventListener('touchmove', (e) => {
        if (e.touches.length > 0) {
            const touch = e.touches[0];
            const mouseX = (touch.clientX / window.innerWidth) * 100;
            const mouseY = (touch.clientY / window.innerHeight) * 100;
            
            document.documentElement.style.setProperty('--mouse-x', `${mouseX}%`);
            document.documentElement.style.setProperty('--mouse-y', `${mouseY}%`);
        }
    });

    // Initialize cursor position
    document.documentElement.style.setProperty('--mouse-x', '50%');
    document.documentElement.style.setProperty('--mouse-y', '50%');
});
</script>
"""

# Replace your existing CSS application with this:
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Mrs+Saint+Delafield&display=swap');
        
        .stApp h2, .stApp .subtitle {{
            font-family: 'Mrs Saint Delafield', cursive !important;
            font-size: 2.5rem !important;
            font-weight: normal !important;
        }}
        
        {enhanced_theme_css}
        
        .stApp h1 {{
            margin-bottom: 0 !important;
            text-align: center !important;
            font-size: 4.5rem !important;
            font-weight: bold !important;
        }}
        .subtitle {{
            margin-top: -15px !important;
            text-align: center !important;
            margin-top: -15px !important;
            padding-top: 0 !important;
            margin-bottom: 1.5rem !important;
        }}
    </style>
    {cursor_tracking_js}
""", unsafe_allow_html=True)

# Also update the data-theme JavaScript
st.markdown(f"""
    <script>
        const app = window.parent.document.querySelector('.stApp');
        if (app) {{
            app.setAttribute('data-theme', '{st.session_state.theme}');
        }}
    </script>
""", unsafe_allow_html=True)

css = """
/* Cursor-following gradient variables */
:root {
    --mouse-x: 50%;
    --mouse-y: 50%;
}

/* Enhanced border gradients with cursor morphing */
.light-mode-border {
    background: linear-gradient(135deg, 
        hsl(330, 80%, 85%) 0%,    /* Light pink */
        hsl(320, 70%, 80%) 25%,   /* Soft pink */
        hsl(310, 75%, 85%) 50%,   /* Rose pink */
        hsl(300, 80%, 90%) 75%,   /* Very light pink */
        hsl(290, 70%, 85%) 100%   /* Lavender pink */
    );
    background-size: 200% 200%;
    animation: lightModeShift 8s ease-in-out infinite;
}

.dark-mode-border {
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        #FF69B4 0%,     /* Hot pink */
        #8A2BE2 20%,    /* Blue violet */
        #4B0082 40%,    /* Indigo */
        #9400D3 60%,    /* Violet */
        #FFE4B5 80%,    /* Moccasin (yellowish) */
        #FF69B4 100%    /* Back to hot pink */
    );
    background-size: 300% 300%;
    animation: darkModeShift 6s ease-in-out infinite;
}

/* Keyframe animations */
@keyframes lightModeShift {
    0%, 100% { 
        background-position: 0% 50%;
        filter: hue-rotate(0deg);
    }
    25% { 
        background-position: 50% 0%;
        filter: hue-rotate(10deg);
    }
    50% { 
        background-position: 100% 50%;
        filter: hue-rotate(0deg);
    }
    75% { 
        background-position: 50% 100%;
        filter: hue-rotate(-10deg);
    }
}

@keyframes darkModeShift {
    0%, 100% { 
        background-position: 0% 50%;
        filter: hue-rotate(0deg) saturate(1.2);
    }
    25% { 
        background-position: 50% 0%;
        filter: hue-rotate(30deg) saturate(1.4);
    }
    50% { 
        background-position: 100% 50%;
        filter: hue-rotate(60deg) saturate(1.6);
    }
    75% { 
        background-position: 50% 100%;
        filter: hue-rotate(30deg) saturate(1.4);
    }
}

/* Cursor-following morphing gradient for main container */
.morphing-gradient-container {
    position: relative;
    overflow: hidden;
}

.morphing-gradient-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        rgba(255, 105, 180, 0.3) 0%,
        rgba(138, 43, 226, 0.2) 25%,
        rgba(75, 0, 130, 0.1) 50%,
        rgba(148, 0, 211, 0.2) 75%,
        rgba(255, 228, 181, 0.1) 100%
    );
    pointer-events: none;
    z-index: -1;
    transition: all 0.3s ease;
}

/* Light mode morphing background overlay */
.light-morphing-gradient::before {
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        rgba(255, 182, 193, 0.4) 0%,    /* Light pink */
        rgba(255, 160, 200, 0.3) 25%,   /* Soft pink */
        rgba(255, 192, 203, 0.2) 50%,   /* Pink */
        rgba(240, 128, 128, 0.3) 75%,   /* Light coral */
        rgba(255, 218, 185, 0.2) 100%   /* Peach puff */
    ) !important;
}

/* Enhanced file uploader with cursor morphing */
.morphing-file-uploader {
    position: relative;
    border-radius: 12px !important;
    padding: 0.5rem !important;
    overflow: hidden;
    transition: all 0.3s ease;
}

.morphing-file-uploader::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    border-radius: 12px;
    padding: 3px;
    background: linear-gradient(
        45deg,
        hsl(calc(var(--mouse-x) * 3.6), 80%, 60%),
        hsl(calc(var(--mouse-y) * 3.6 + 60), 75%, 65%),
        hsl(calc(var(--mouse-x) * 3.6 + 120), 70%, 70%),
        hsl(calc(var(--mouse-y) * 3.6 + 180), 80%, 60%)
    );
    background-size: 400% 400%;
    animation: gradientShift 4s ease infinite;
    z-index: -1;
}

.morphing-file-uploader::after {
    content: '';
    position: absolute;
    top: 3px;
    left: 3px;
    right: 3px;
    bottom: 3px;
    background: var(--bg-color);
    border-radius: 9px;
    z-index: -1;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    25% { background-position: 100% 0%; }
    50% { background-position: 100% 100%; }
    75% { background-position: 0% 100%; }
}

/* Theme toggle button with morphing effect */
.morphing-theme-toggle {
    position: relative;
    border-radius: 50% !important;
    width: 50px !important;
    height: 50px !important;
    overflow: hidden;
}

.morphing-theme-toggle::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(
        from calc(var(--mouse-x) * 3.6deg),
        #FF69B4, #8A2BE2, #4B0082, #9400D3, #FFE4B5, #FF69B4
    );
    animation: rotateMorph 8s linear infinite;
    z-index: -1;
}

@keyframes rotateMorph {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Interactive gradient cards */
.gradient-card {
    position: relative;
    border-radius: 12px;
    padding: 20px;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: pointer;
}

.gradient-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(
        circle at var(--mouse-x) var(--mouse-y),
        rgba(255, 105, 180, 0.1) 0%,
        rgba(138, 43, 226, 0.05) 50%,
        rgba(75, 0, 130, 0.1) 100%
    );
    opacity: 0;
    transition: opacity 0.3s ease;
    z-index: -1;
}

.gradient-card:hover::before {
    opacity: 1;
}

/* Responsive cursor tracking */
@media (max-width: 768px) {
    .morphing-gradient-container::before,
    .gradient-card::before {
        background-size: 400% 400%;
        animation: mobileMorph 6s ease-in-out infinite;
    }
}

@keyframes mobileMorph {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
```

# JavaScript for Cursor Tracking (add this to your Streamlit app):

```javascript
// Cursor tracking for morphing gradients
document.addEventListener('mousemove', (e) => {
    const mouseX = (e.clientX / window.innerWidth) * 100;
    const mouseY = (e.clientY / window.innerHeight) * 100;
    
    document.documentElement.style.setProperty('--mouse-x', `${mouseX}%`);
    document.documentElement.style.setProperty('--mouse-y', `${mouseY}%`);
});

// Enhanced touch support for mobile
document.addEventListener('touchmove', (e) => {
    if (e.touches.length > 0) {
        const touch = e.touches[0];
        const mouseX = (touch.clientX / window.innerWidth) * 100;
        const mouseY = (touch.clientY / window.innerHeight) * 100;
        
        document.documentElement.style.setProperty('--mouse-x', `${mouseX}%`);
        document.documentElement.style.setProperty('--mouse-y', `${mouseY}%`);
    }
});

// Smooth transition on page load
window.addEventListener('load', () => {
    document.documentElement.style.setProperty('--mouse-x', '50%');
    document.documentElement.style.setProperty('--mouse-y', '50%');
});
"""

# Add this JavaScript to set the data-theme attribute
st.markdown(f"""
    <script>
        // Set data-theme attribute based on session state theme
        const app = window.parent.document.querySelector('.stApp');
        if (app) {{
            app.setAttribute('data-theme', '{st.session_state.theme}');
        }}
    </script>
""", unsafe_allow_html=True)

# Title and description with animation
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
st.title("AutoML Solution")
st.markdown("<h2 class='subtitle'>Just upload your dataset, select parameters, and train the models and see the Magic!</h2>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Navigation subtitle style with reduced size
st.markdown("""
    <style>
        .stApp .stSidebar .subtitle {{
            font-size: 1.2rem !important;  /* Reduced from previous 2.0rem/1.8rem */
        }}
    </style>
""", unsafe_allow_html=True)

# Main page navigation under the title
st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
page = st.radio(
    "Navigation",
    ["Data Upload", "Model Selection", "Training", "Results"],
    horizontal=True,
    index=0,
    key="main_nav"
)
st.markdown("</div>", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'model_selected' not in st.session_state:
    st.session_state.model_selected = False
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'param_grid' not in st.session_state:
    st.session_state.param_grid = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None

# Data Upload Page
if page == "Data Upload":
    
    # Create a pulsing upload area with theme-specific styling
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and display the data
        df = pd.read_csv("temp_data.csv")
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Add data analysis section
        st.subheader("Data Analysis")
        
        # Display basic statistics
        st.write("Basic Statistics:")
        st.write(df.describe())
        
        # Check for missing values
        st.write("Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.any() > 0 else "No missing values")
        
        # Check class distribution if target column exists
        st.write("Select target column for analysis:")
        target_column = st.selectbox("Select the target column", df.columns)
        
        if target_column:
            st.write("Target Distribution:")
            target_counts = df[target_column].value_counts()
            st.write(target_counts)
            
            # Visualize class distribution
            st.bar_chart(target_counts)
            
            # Check if highly imbalanced
            if len(target_counts) > 1:
                imbalance_ratio = target_counts.max() / target_counts.min()
                if imbalance_ratio > 10:
                    st.warning(f"Warning: Highly imbalanced classes detected (ratio: {imbalance_ratio:.2f}). Consider using class weights or resampling techniques.")
    
    # Select test size with a more theme-consistent slider
    st.write("Select test size for train-test split:")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    
    # Create a more attractive preprocess button
    preprocess_button_style = f"""
    <style>
    div[data-testid="stButton"]:has(button:contains("Preprocess Data")) button {{
        background: {light_mode_border if st.session_state.theme == 'light' else dark_mode_border} !important;
        color: {'#000000' if st.session_state.theme == 'light' else '#FFFFFF'} !important;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    div[data-testid="stButton"]:has(button:contains("Preprocess Data")) button:hover {{
        transform: scale(1.05);
        box-shadow: 0 0 15px {'rgba(115, 3, 192, 0.7)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.7)'};
    }}
    </style>
    """
    st.markdown(preprocess_button_style, unsafe_allow_html=True)
    
    if st.button("Preprocess Data"):
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
        elif not target_column:
            st.error("Please select a target column.")
        else:
            try:
                with st.spinner("Preprocessing data... Please wait."):
                    X_train, X_test, y_train, y_test, preprocessor = preprocess_data("temp_data.csv", target_column, test_size)
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.preprocessor = preprocessor
                    st.session_state.data_uploaded = True
                
                # Create success message with animation
                success_style = f"""
                <style>
                .success-message {{
                    padding: 15px;
                    border-radius: 8px;
                    background-color: {'rgba(76, 175, 80, 0.1)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.1)'};
                    border-left: 5px solid {'#4CAF50' if st.session_state.theme == 'light' else '#FF69B4'};
                    margin: 20px 0;
                    animation: fadeIn 0.5s ease-in;
                }}
                </style>
                <div class="success-message">
                    <b>✅ Data preprocessing completed successfully!</b>
                    <p>Training set shape: {X_train.shape}</p>
                    <p>Test set shape: {X_test.shape}</p>
                </div>
                """
                st.markdown(success_style, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred during preprocessing: {e}")

# Model Selection Page
elif page == "Model Selection":
    if not st.session_state.data_uploaded:
        st.warning("Please upload and preprocess your data first!")
    else:
        st.header("Select Machine Learning Algorithm")
        
        algorithms = [
            'SVM',
            'Decision Tree',
            'Bagging',
            'Random Forest',
            'ADA Boost',
            'XG Boost',
            'Neural Network'
        ]
        
        # Create a more attractive algorithm selector
        model_name = st.selectbox("Choose an algorithm", algorithms)
        st.session_state.model_name = model_name
        
        # Display algorithm information
        algorithm_info = {
            'SVM': "Support Vector Machine works well for complex but small or medium-sized datasets.",
            'Decision Tree': "Simple to understand and visualize, but can easily overfit.",
            'Bagging': "Uses multiple instances of a base estimator to reduce variance.",
            'Random Forest': "An ensemble of decision trees that works well for many problems.",
            'ADA Boost': "Focuses on difficult samples by boosting their importance.",
            'XG Boost': "Gradient boosting implementation known for its speed and performance.",
            'Neural Network': "Can capture complex patterns but might require more data."
        }
        
        # Show algorithm info in a styled box
        if model_name in algorithm_info:
            info_style = f"""
            <style>
            .algorithm-info {{
                padding: 15px;
                border-radius: 8px;
                background-color: {'rgba(115, 3, 192, 0.05)' if st.session_state.theme == 'light' else 'rgba(255, 105, 180, 0.05)'};
                border-left: 5px solid {'#7303c0' if st.session_state.theme == 'light' else '#FF69B4'};
                margin-bottom: 20px;
            }}
            </style>
            <div class="algorithm-info">
                <p>{algorithm_info[model_name]}</p>
            </div>
            """
            st.markdown(info_style, unsafe_allow_html=True)
        
        # Hyperparameters section with improved styling
        st.subheader(f"Set Hyperparameters for {model_name}")
        
        # Create a styled hyperparameter container
        hyperparam_style = f"""
        <style>
        .hyperparam-container {{
            padding: 15px;
            border-radius: 8px;
            background-color: {'rgba(0, 0, 255, 0.05)' if st.session_state.theme == 'light' else 'rgba(178, 141, 255, 0.05)'};
            border: 1px solid {'rgba(0, 0, 255, 0.2)' if st.session_state.theme == 'light' else 'rgba(178, 141, 255, 0.2)'};
            margin-bottom: 20px;
        }}
        </style>
        <div class="hyperparam-container">
        """
        st.markdown(hyperparam_style, unsafe_allow_html=True)
        
        param_grid = {}
        
        if model_name == 'SVM':
            param_grid['C'] = [float(c) for c in st.text_input("Values for C (comma separated)", "0.1,1,10").split(',')]
            param_grid['kernel'] = st.multiselect("Kernel", ['linear', 'rbf', 'poly', 'sigmoid'], ['linear', 'rbf'])
            param_grid['gamma'] = st.multiselect("Gamma", ['scale', 'auto'], ['scale'])
            
        elif model_name == 'Decision Tree':
            param_grid['criterion'] = st.multiselect("Criterion", ['gini', 'entropy'], ['gini'])
            param_grid['max_depth'] = [int(d) if d.strip() else None for d in st.text_input("Values for max_depth (comma separated, e.g., 10,20,None)", "10,20,30").split(',')]
            param_grid['max_depth'] = [d for d in param_grid['max_depth'] if d is not None] # Filter out Nones if user meant to skip
            if not param_grid['max_depth']: param_grid['max_depth'] = [None] # Default to None if empty

            param_grid['min_samples_split'] = [int(s) for s in st.text_input("Values for min_samples_split (comma separated)", "2,5,10").split(',')]
            param_grid['min_samples_leaf'] = [int(l) for l in st.text_input("Values for min_samples_leaf (comma separated)", "1,2,4").split(',')]
            
        elif model_name == 'Bagging':
            param_grid['n_estimators'] = [int(n) for n in st.text_input("Values for n_estimators (comma separated)", "10,50,100").split(',')]
            param_grid['max_samples'] = [float(s) for s in st.text_input("Values for max_samples (comma separated)", "0.5,0.7,1.0").split(',')]
            param_grid['max_features'] = [float(f) for f in st.text_input("Values for max_features (comma separated)", "0.5,0.7,1.0").split(',')]
            
        elif model_name == 'Random Forest':
            param_grid['n_estimators'] = [int(n) for n in st.text_input("Values for n_estimators (comma separated)", "50,100,200").split(',')]
            param_grid['max_depth'] = [int(d) if d.strip() else None for d in st.text_input("Values for max_depth (comma separated, e.g., 10,20,None)", "10,20,30").split(',')]
            param_grid['max_depth'] = [d for d in param_grid['max_depth'] if d is not None]
            if not param_grid['max_depth']: param_grid['max_depth'] = [None]

            param_grid['min_samples_split'] = [int(s) for s in st.text_input("Values for min_samples_split (comma separated)", "2,5,10").split(',')]
            param_grid['min_samples_leaf'] = [int(l) for l in st.text_input("Values for min_samples_leaf (comma separated)", "1,2,4").split(',')]
            param_grid['max_features'] = st.multiselect("Max features", ['sqrt', 'log2', None], ['sqrt'])
            
        elif model_name == 'ADA Boost':
            param_grid['n_estimators'] = [int(n) for n in st.text_input("Values for n_estimators (comma separated)", "50,100,200").split(',')]
            param_grid['learning_rate'] = [float(r) for r in st.text_input("Values for learning_rate (comma separated)", "0.01,0.1,1.0").split(',')]
            
        elif model_name == 'XG Boost':
            param_grid['n_estimators'] = [int(n) for n in st.text_input("Values for n_estimators (comma separated)", "50,100,200").split(',')]
            param_grid['learning_rate'] = [float(r) for r in st.text_input("Values for learning_rate (comma separated)", "0.01,0.1,0.2").split(',')]
            param_grid['max_depth'] = [int(d) for d in st.text_input("Values for max_depth (comma separated)", "3,6,10").split(',')]
            param_grid['subsample'] = [float(s) for s in st.text_input("Values for subsample (comma separated)", "0.6,0.8,1.0").split(',')]
            
        # In the Model Selection page, update the Neural Network parameter handling:
        elif model_name == 'Neural Network':
            hidden_layer_input = st.text_input("Values for hidden_layer_sizes (comma separated, e.g., 50,100)", "50,100")
            hidden_layer_sizes = []
            for layer in hidden_layer_input.split(','):
                try:
                    # Convert to integer and create a tuple
                    hidden_layer_sizes.append((int(layer.strip()),))
                except ValueError:
                    st.error(f"Invalid value for hidden layer: {layer}")
            param_grid['hidden_layer_sizes'] = hidden_layer_sizes
            param_grid['alpha'] = [float(a) for a in st.text_input("Values for alpha (comma separated)", "0.0001,0.001,0.01").split(',')]
            param_grid['activation'] = st.multiselect("Activation", ['relu', 'tanh', 'logistic'], ['relu'])
            param_grid['solver'] = st.multiselect("Solver", ['adam', 'sgd', 'lbfgs'], ['adam'])
            param_grid['learning_rate_init'] = [float(lr) for lr in st.text_input("Initial Learning Rates (comma separated, for sgd/adam)", "0.001,0.01").split(',')]
        
        if st.button("Confirm Model Selection"):
            st.session_state.param_grid = param_grid
            st.session_state.model_selected = True
            st.success(f"Model {model_name} selected with hyperparameters!")
            st.write("Selected Hyperparameters for Search:")
            st.json(param_grid)


# Training Page
elif page == "Training":
    if not st.session_state.data_uploaded:
        st.warning("Please upload and preprocess your data first!")
    elif not st.session_state.model_selected:
        st.warning("Please select a model and set hyperparameters first!")
    else:
        st.header("Train Your Model")
        
        st.write(f"Selected Model: {st.session_state.model_name}")
        st.write("Hyperparameters for Search:")
        st.write(st.session_state.param_grid)
        
        if st.button("Train Model"):
            with st.spinner("Training in progress... This may take a while."):
                try:
                    # Get the base model
                    model_instance = None
                    if st.session_state.model_name == 'SVM':
                        model_instance = SVC(class_weight='balanced', probability=True)
                    elif st.session_state.model_name == 'Decision Tree':
                        model_instance = DecisionTreeClassifier(class_weight='balanced')
                    elif st.session_state.model_name == 'Bagging':
                        model_instance = BaggingClassifier()
                    elif st.session_state.model_name == 'Random Forest':
                        model_instance = RandomForestClassifier(class_weight='balanced')
                    elif st.session_state.model_name == 'ADA Boost':
                        model_instance = AdaBoostClassifier()
                    elif st.session_state.model_name == 'XG Boost':
                        model_instance = XGBClassifier(use_label_encoder=False, eval_metric='logloss') # Suppress warning and set eval_metric
                    elif st.session_state.model_name == 'Neural Network':
                        model_instance = MLPClassifier(max_iter=1000, early_stopping=True)
                    
                    param_grid_to_use = st.session_state.param_grid
                    
                    # Adjust for class imbalance if needed for XGBoost
                    if st.session_state.model_name == 'XG Boost':
                        unique_classes, counts = np.unique(st.session_state.y_train, return_counts=True)
                        if len(unique_classes) == 2: # Binary classification
                            weight_ratio = counts[0] / counts[1] if counts[1] > 0 else 1
                            # XGBoost doesn't have scale_pos_weight in param_grid, it's a fit param or init param
                            # So, we can't directly put it in param_grid for RandomizedSearchCV
                            # Instead, it's often handled by setting it during model initialization if not part of grid search
                            # For simplicity, we'll rely on RandomizedSearchCV to find good params.
                            # If scale_pos_weight is crucial, it might need a custom loop or be set if not in param_grid.
                            # Current XGBClassifier init does not include scale_pos_weight. It can be added:
                            # model_instance = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=weight_ratio)
                            # However, this would apply it always, not as part of the search.
                            # For now, we assume class_weight='balanced' equivalent strategies in other models help.

                    # Calculate n_iter for RandomizedSearchCV
                    # Total combinations: multiply lengths of lists in param_grid
                    total_combinations = 1
                    for key in param_grid_to_use:
                        total_combinations *= len(param_grid_to_use[key])
                    
                    n_iter_val = min(10, total_combinations) # Default to 10 iterations or fewer if less combinations

                    search = RandomizedSearchCV(
                        model_instance, 
                        param_grid_to_use, 
                        n_iter=n_iter_val, 
                        cv=3, # Reduced CV for faster runs during hyperparameter tuning
                        random_state=42, 
                        n_jobs=1, 
                        scoring='f1_weighted'
                    )
                    
                    search.fit(st.session_state.X_train, st.session_state.y_train)
                    best_model = search.best_estimator_
                    
                    y_pred = best_model.predict(st.session_state.X_test)
                    
                    metrics = {
                        'accuracy': accuracy_score(st.session_state.y_test, y_pred),
                        'precision': precision_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                    }
                    
                    st.session_state.metrics = metrics
                    st.session_state.best_model = best_model
                    st.session_state.training_done = True
                    
                    if not os.path.exists("models"):
                        os.makedirs("models")
                    joblib.dump(best_model, f"models/{st.session_state.model_name}_model.pkl")
                    joblib.dump(st.session_state.preprocessor, "models/preprocessor.pkl")
                    
                    st.success("Training completed successfully!")
                    st.write("Best Parameters Found:")
                    st.json(search.best_params_)
                    
                except Exception as e:
                    st.error(f"An error occurred during training: {e}")
                    import traceback
                    st.error(traceback.format_exc())


# Results Page
elif page == "Results":
    if not st.session_state.training_done:
        st.warning("Please train a model first!")
    else:
        st.header("Model Evaluation Results")
        
        st.subheader("Performance Metrics")
        metrics = st.session_state.metrics
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("F1 Score (Weighted)", f"{metrics['f1']:.4f}")
        
        st.subheader("Best Hyperparameters")
        # st.json(st.session_state.best_model.get_params()) # This can be very verbose
        # Displaying search.best_params_ is often cleaner if available from training step
        # For now, let's show get_params()
        try:
            best_params_display = {k: v for k, v in st.session_state.best_model.get_params().items() if k in st.session_state.param_grid or k in ['n_estimators', 'C', 'kernel']} # Show relevant params
            st.json(best_params_display)
        except Exception as e:
            st.write("Could not display all best parameters.")


        st.subheader("Classification Report")
        try:
            y_pred_report = st.session_state.best_model.predict(st.session_state.X_test)
            report = classification_report(st.session_state.y_test, y_pred_report, output_dict=False, zero_division=0)
            st.text_area("Report", report, height=250)
        except Exception as e:
            st.error(f"Could not generate classification report: {e}")

        
        st.subheader("Download Trained Model and Preprocessor")
        
        try:
            with open(f"models/{st.session_state.model_name}_model.pkl", "rb") as fp:
                st.download_button(
                    label=f"Download {st.session_state.model_name}_model.pkl",
                    data=fp,
                    file_name=f"{st.session_state.model_name}_model.pkl",
                    mime="application/octet-stream"
                )
            with open("models/preprocessor.pkl", "rb") as fp:
                st.download_button(
                    label="Download preprocessor.pkl",
                    data=fp,
                    file_name="preprocessor.pkl",
                    mime="application/octet-stream"
                )
        except FileNotFoundError:
            st.error("Model files not found. Please train the model again.")
        except Exception as e:
            st.error(f"Error preparing download: {e}")

        
        st.subheader("Make New Predictions")
        st.write("Upload a new CSV file with the same features (without target column) as your training data to make predictions.")
        
        new_file = st.file_uploader("Choose a CSV file for prediction", type="csv", key="new_pred_file")
        
        if new_file is not None:
            temp_pred_file_path = "temp_pred_data.csv"
            with open(temp_pred_file_path, "wb") as f:
                f.write(new_file.getbuffer())
            
            new_df = pd.read_csv(temp_pred_file_path)
            st.write("New Data Preview:")
            st.dataframe(new_df.head())
            
            if st.button("Make Predictions on New Data"):
                try:
                    # Ensure preprocessor and model are loaded from session state
                    if st.session_state.preprocessor and st.session_state.best_model:
                        new_data_transformed = st.session_state.preprocessor.transform(new_df)
                        predictions = st.session_state.best_model.predict(new_data_transformed)
                        
                        new_df_with_predictions = new_df.copy()
                        new_df_with_predictions['Prediction'] = predictions
                        
                        st.write("Predictions:")
                        st.dataframe(new_df_with_predictions)
                        
                        csv_predictions = new_df_with_predictions.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_predictions,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Preprocessor or model not found in session. Please ensure training was successful.")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    import traceback
                    st.error(traceback.format_exc())
            
            # Clean up temporary prediction file
            if os.path.exists(temp_pred_file_path):
                os.remove(temp_pred_file_path)
