"""
Clean, Professional Loyalty Engine Demo
Interview-ready 3-tab interface showcasing ML capabilities
"""

import streamlit as st
import os
import sys

# Add frontend_tabs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend_tabs'))

# Import with error handling
try:
    from frontend_tabs.classic_ml_tab import run_classic_ml_tab
    from frontend_tabs.adaptive_ai_tab import run_adaptive_ai_tab
    from frontend_tabs.comparison_tab import run_comparison_tab
    from frontend_tabs.shared_components import apply_professional_theme, display_demo_instructions
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Loyalty Engine - ML Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply professional styling
apply_professional_theme()

# Main header
st.title("🎯 Coffee Shop Loyalty Engine")
st.markdown("""
**Advanced ML System for Customer Loyalty Optimization**  
Demonstrating traditional ML vs adaptive AI approaches for promotion targeting
""")

# Demo instructions (collapsible)
display_demo_instructions()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🎯 Classic ML",
    "🧠 Adaptive AI", 
    "⚖️ Comparison"
])

# Tab 1: Classic ML
with tab1:
    try:
        run_classic_ml_tab()
    except Exception as e:
        st.error(f"Error in Classic ML tab: {str(e)}")
        st.info("Try generating data first: Click 'Generate Data' button")

# Tab 2: Adaptive AI
with tab2:
    try:
        run_adaptive_ai_tab()
    except Exception as e:
        st.error(f"Error in Adaptive AI tab: {str(e)}")
        st.info("The Adaptive AI system may need initialization")

# Tab 3: Comparison
with tab3:
    try:
        run_comparison_tab()
    except Exception as e:
        st.error(f"Error in Comparison tab: {str(e)}")
        st.info("Run both Classic ML and Adaptive AI first to enable comparison")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
Built with Python | Scikit-learn | River ML | Streamlit<br>
Ready for production deployment at scale
</div>
""", unsafe_allow_html=True)