"""
Simplest entry-point for launching the Streamlit AutoML UI.
Execute:  streamlit run run.py
"""

from src.ui.streamlit_app import main

# Streamlit only needs the call below; everything else
# is handled by the Streamlit CLI.
if __name__ == "__main__":
    main()
